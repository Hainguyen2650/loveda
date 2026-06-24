#define _POSIX_C_SOURCE 200809L
#define STB_IMAGE_IMPLEMENTATION

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../third_party/stb_image.h"

#define GROUP_COUNT 6
#define CHANNEL_COUNT 3
#define HIST_SIZE 256
#define MOD_COUNT 3

typedef struct {
    const char *split;
    const char *domain;
    const char *rel_dir;
} GroupInfo;

static const GroupInfo GROUPS[GROUP_COUNT] = {
    {"Train", "Urban", "Train/Urban/images_png"},
    {"Train", "Rural", "Train/Rural/images_png"},
    {"Val", "Urban", "Val/Urban/images_png"},
    {"Val", "Rural", "Val/Rural/images_png"},
    {"Test", "Urban", "Test/Urban/images_png"},
    {"Test", "Rural", "Test/Rural/images_png"},
};

static const char *CHANNELS[CHANNEL_COUNT] = {"R", "G", "B"};
static const int MODULI[MOD_COUNT] = {2, 4, 8};

typedef struct {
    char *path;
    int group_id;
} FileTask;

typedef struct {
    uint64_t sample_count[GROUP_COUNT];
    uint64_t total_pixel_count[GROUP_COUNT];
    uint64_t pixel_count[GROUP_COUNT];
    uint64_t black_pixel_count[GROUP_COUNT];
    uint64_t images_with_black[GROUP_COUNT];
    long double channel_sum[GROUP_COUNT][CHANNEL_COUNT];
    long double channel_sumsq[GROUP_COUNT][CHANNEL_COUNT];
    uint8_t channel_min[GROUP_COUNT][CHANNEL_COUNT];
    uint8_t channel_max[GROUP_COUNT][CHANNEL_COUNT];
    uint64_t unique_value_sum[GROUP_COUNT][CHANNEL_COUNT];
    uint64_t histogram[GROUP_COUNT][CHANNEL_COUNT][HIST_SIZE];
} ThreadStats;

typedef struct {
    FileTask *tasks;
    size_t task_count;
    size_t next_index;
    pthread_mutex_t mutex;
} TaskQueue;

typedef struct {
    TaskQueue *queue;
    ThreadStats *stats;
} WorkerArgs;

static void init_stats(ThreadStats *stats) {
    memset(stats, 0, sizeof(*stats));
    for (int g = 0; g < GROUP_COUNT; ++g) {
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            stats->channel_min[g][c] = 255;
            stats->channel_max[g][c] = 0;
        }
    }
}

static int cmp_tasks(const void *lhs, const void *rhs) {
    const FileTask *a = (const FileTask *)lhs;
    const FileTask *b = (const FileTask *)rhs;
    int group_cmp = a->group_id - b->group_id;
    if (group_cmp != 0) {
        return group_cmp;
    }
    return strcmp(a->path, b->path);
}

static bool has_png_suffix(const char *name) {
    size_t len = strlen(name);
    return len >= 4 && strcmp(name + len - 4, ".png") == 0;
}

static char *join_path(const char *base, const char *suffix) {
    size_t base_len = strlen(base);
    size_t suffix_len = strlen(suffix);
    bool need_slash = base_len > 0 && base[base_len - 1] != '/';
    size_t total = base_len + suffix_len + (need_slash ? 2 : 1);
    char *buf = (char *)malloc(total);
    if (buf == NULL) {
        return NULL;
    }
    snprintf(buf, total, need_slash ? "%s/%s" : "%s%s", base, suffix);
    return buf;
}

static int ensure_dir(const char *path) {
    char *tmp = strdup(path);
    if (tmp == NULL) {
        return -1;
    }

    size_t len = strlen(tmp);
    if (len == 0) {
        free(tmp);
        return 0;
    }

    for (size_t i = 1; i < len; ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                free(tmp);
                return -1;
            }
            tmp[i] = '/';
        }
    }

    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        free(tmp);
        return -1;
    }

    free(tmp);
    return 0;
}

static bool collect_group_files(
    const char *dataset_root,
    int group_id,
    FileTask **tasks,
    size_t *task_count,
    size_t *task_cap,
    long max_images_per_group
) {
    char *dir_path = join_path(dataset_root, GROUPS[group_id].rel_dir);
    if (dir_path == NULL) {
        fprintf(stderr, "Failed to allocate directory path\n");
        return false;
    }

    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        free(dir_path);
        return true;
    }

    long collected = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (!has_png_suffix(entry->d_name)) {
            continue;
        }
        if (max_images_per_group > 0 && collected >= max_images_per_group) {
            break;
        }
        if (*task_count == *task_cap) {
            size_t new_cap = *task_cap == 0 ? 1024 : *task_cap * 2;
            FileTask *new_tasks = (FileTask *)realloc(*tasks, new_cap * sizeof(FileTask));
            if (new_tasks == NULL) {
                fprintf(stderr, "Failed to grow task list\n");
                closedir(dir);
                free(dir_path);
                return false;
            }
            *tasks = new_tasks;
            *task_cap = new_cap;
        }
        char *file_path = join_path(dir_path, entry->d_name);
        if (file_path == NULL) {
            fprintf(stderr, "Failed to allocate file path\n");
            closedir(dir);
            free(dir_path);
            return false;
        }
        (*tasks)[*task_count].path = file_path;
        (*tasks)[*task_count].group_id = group_id;
        *task_count += 1;
        collected += 1;
    }

    closedir(dir);
    free(dir_path);
    return true;
}

static void update_channel_stats(ThreadStats *stats, int group_id, const unsigned char *pixels, int width, int height) {
    uint64_t pixel_total = (uint64_t)width * (uint64_t)height;
    bool seen[CHANNEL_COUNT][HIST_SIZE];
    bool has_black = false;
    memset(seen, 0, sizeof(seen));

    stats->sample_count[group_id] += 1;
    stats->total_pixel_count[group_id] += pixel_total;

    const unsigned char *ptr = pixels;
    uint64_t count = pixel_total;
    while (count-- > 0) {
        if (ptr[0] == 0 && ptr[1] == 0 && ptr[2] == 0) {
            stats->black_pixel_count[group_id] += 1;
            has_black = true;
            ptr += CHANNEL_COUNT;
            continue;
        }
        stats->pixel_count[group_id] += 1;
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            uint8_t value = ptr[c];
            stats->channel_sum[group_id][c] += (long double)value;
            stats->channel_sumsq[group_id][c] += (long double)value * (long double)value;
            if (value < stats->channel_min[group_id][c]) {
                stats->channel_min[group_id][c] = value;
            }
            if (value > stats->channel_max[group_id][c]) {
                stats->channel_max[group_id][c] = value;
            }
            stats->histogram[group_id][c][value] += 1;
            seen[c][value] = true;
        }
        ptr += CHANNEL_COUNT;
    }

    if (has_black) {
        stats->images_with_black[group_id] += 1;
    }

    for (int c = 0; c < CHANNEL_COUNT; ++c) {
        for (int i = 0; i < HIST_SIZE; ++i) {
            if (seen[c][i]) {
                stats->unique_value_sum[group_id][c] += 1;
            }
        }
    }
}

static void *worker_main(void *arg) {
    WorkerArgs *worker = (WorkerArgs *)arg;

    for (;;) {
        pthread_mutex_lock(&worker->queue->mutex);
        size_t index = worker->queue->next_index;
        if (index >= worker->queue->task_count) {
            pthread_mutex_unlock(&worker->queue->mutex);
            break;
        }
        worker->queue->next_index += 1;
        FileTask task = worker->queue->tasks[index];
        pthread_mutex_unlock(&worker->queue->mutex);

        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char *pixels = stbi_load(task.path, &width, &height, &channels, CHANNEL_COUNT);
        if (pixels == NULL) {
            fprintf(stderr, "Failed to load %s: %s\n", task.path, stbi_failure_reason());
            continue;
        }

        update_channel_stats(worker->stats, task.group_id, pixels, width, height);
        stbi_image_free(pixels);
    }

    return NULL;
}

static void merge_stats(ThreadStats *dst, const ThreadStats *src) {
    for (int g = 0; g < GROUP_COUNT; ++g) {
        dst->sample_count[g] += src->sample_count[g];
        dst->total_pixel_count[g] += src->total_pixel_count[g];
        dst->pixel_count[g] += src->pixel_count[g];
        dst->black_pixel_count[g] += src->black_pixel_count[g];
        dst->images_with_black[g] += src->images_with_black[g];
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            dst->channel_sum[g][c] += src->channel_sum[g][c];
            dst->channel_sumsq[g][c] += src->channel_sumsq[g][c];
            if (src->pixel_count[g] > 0) {
                if (src->channel_min[g][c] < dst->channel_min[g][c]) {
                    dst->channel_min[g][c] = src->channel_min[g][c];
                }
                if (src->channel_max[g][c] > dst->channel_max[g][c]) {
                    dst->channel_max[g][c] = src->channel_max[g][c];
                }
            }
            dst->unique_value_sum[g][c] += src->unique_value_sum[g][c];
            for (int i = 0; i < HIST_SIZE; ++i) {
                dst->histogram[g][c][i] += src->histogram[g][c][i];
            }
        }
    }
}

static long double mean_abs_first_diff(const uint64_t hist[HIST_SIZE]) {
    long double total = 0.0L;
    for (int i = 0; i < HIST_SIZE; ++i) {
        total += (long double)hist[i];
    }
    if (total <= 0.0L) {
        return 0.0L;
    }

    long double accum = 0.0L;
    for (int i = 0; i < HIST_SIZE - 1; ++i) {
        long double lhs = (long double)hist[i] / total;
        long double rhs = (long double)hist[i + 1] / total;
        accum += fabsl(rhs - lhs);
    }
    return accum / (long double)(HIST_SIZE - 1);
}

static long double mean_abs_second_diff(const uint64_t hist[HIST_SIZE]) {
    long double total = 0.0L;
    for (int i = 0; i < HIST_SIZE; ++i) {
        total += (long double)hist[i];
    }
    if (total <= 0.0L) {
        return 0.0L;
    }

    long double accum = 0.0L;
    for (int i = 0; i < HIST_SIZE - 2; ++i) {
        long double a = (long double)hist[i] / total;
        long double b = (long double)hist[i + 1] / total;
        long double c = (long double)hist[i + 2] / total;
        accum += fabsl(c - 2.0L * b + a);
    }
    return accum / (long double)(HIST_SIZE - 2);
}

static long double parity_bias(const uint64_t hist[HIST_SIZE]) {
    long double even = 0.0L;
    long double odd = 0.0L;
    for (int i = 0; i < HIST_SIZE; ++i) {
        if (i % 2 == 0) {
            even += (long double)hist[i];
        } else {
            odd += (long double)hist[i];
        }
    }
    long double total = even + odd;
    if (total <= 0.0L) {
        return 0.0L;
    }
    return fabsl(even - odd) / total;
}

static long double peak_to_trough_ratio(const uint64_t hist[HIST_SIZE], int low, int high) {
    long double min_positive = -1.0L;
    long double max_value = 0.0L;
    for (int i = low; i <= high; ++i) {
        long double value = (long double)hist[i];
        if (value > 0.0L && (min_positive < 0.0L || value < min_positive)) {
            min_positive = value;
        }
        if (value > max_value) {
            max_value = value;
        }
    }
    if (min_positive <= 0.0L) {
        return 0.0L;
    }
    return max_value / min_positive;
}

static void modulo_profile(const uint64_t hist[HIST_SIZE], int modulus, long double *out) {
    for (int i = 0; i < modulus; ++i) {
        out[i] = 0.0L;
    }

    long double total = 0.0L;
    for (int i = 0; i < HIST_SIZE; ++i) {
        total += (long double)hist[i];
        out[i % modulus] += (long double)hist[i];
    }
    if (total <= 0.0L) {
        return;
    }
    for (int i = 0; i < modulus; ++i) {
        out[i] /= total;
    }
}

static long double modulo_uniformity_score(const long double *profile, int modulus) {
    long double uniform = 1.0L / (long double)modulus;
    long double score = 0.0L;
    for (int i = 0; i < modulus; ++i) {
        score += fabsl(profile[i] - uniform);
    }
    return score;
}

static int dominant_residue(const long double *profile, int modulus) {
    int best_idx = 0;
    long double best_val = profile[0];
    for (int i = 1; i < modulus; ++i) {
        if (profile[i] > best_val) {
            best_val = profile[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static bool write_global_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "rgb_global_stats.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,channel,sample_count,pixel_count,mean,std,min,max\n");
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (stats->sample_count[g] == 0 || stats->pixel_count[g] == 0) {
            continue;
        }
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            long double pixel_count = (long double)stats->pixel_count[g];
            long double mean = stats->channel_sum[g][c] / pixel_count;
            long double var = stats->channel_sumsq[g][c] / pixel_count - mean * mean;
            if (var < 0.0L) {
                var = 0.0L;
            }
            long double std = sqrtl(var);
            fprintf(
                fp,
                "%s,%s,%s,%" PRIu64 ",%" PRIu64 ",%.6Lf,%.6Lf,%u,%u\n",
                GROUPS[g].split,
                GROUPS[g].domain,
                CHANNELS[c],
                stats->sample_count[g],
                stats->pixel_count[g],
                mean,
                std,
                (unsigned int)stats->channel_min[g][c],
                (unsigned int)stats->channel_max[g][c]
            );
        }
    }

    fclose(fp);
    return true;
}

static bool write_histogram_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "rgb_histograms.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,channel,intensity,pixel_count\n");
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (stats->sample_count[g] == 0) {
            continue;
        }
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            for (int i = 0; i < HIST_SIZE; ++i) {
                fprintf(
                    fp,
                    "%s,%s,%s,%d,%" PRIu64 "\n",
                    GROUPS[g].split,
                    GROUPS[g].domain,
                    CHANNELS[c],
                    i,
                    stats->histogram[g][c][i]
                );
            }
        }
    }

    fclose(fp);
    return true;
}

static bool write_quantization_summary_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "quantization_channel_summary.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(
        fp,
        "split,domain,channel,sample_count,pixel_count,mean_unique_values_per_image,"
        "mean_abs_first_diff,mean_abs_second_diff,parity_bias,midrange_peak_to_trough_ratio_40_100\n"
    );
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (stats->sample_count[g] == 0 || stats->pixel_count[g] == 0) {
            continue;
        }
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            long double mean_unique = (long double)stats->unique_value_sum[g][c] / (long double)stats->sample_count[g];
            fprintf(
                fp,
                "%s,%s,%s,%" PRIu64 ",%" PRIu64 ",%.4Lf,%.10Lf,%.10Lf,%.10Lf,%.6Lf\n",
                GROUPS[g].split,
                GROUPS[g].domain,
                CHANNELS[c],
                stats->sample_count[g],
                stats->pixel_count[g],
                mean_unique,
                mean_abs_first_diff(stats->histogram[g][c]),
                mean_abs_second_diff(stats->histogram[g][c]),
                parity_bias(stats->histogram[g][c]),
                peak_to_trough_ratio(stats->histogram[g][c], 40, 100)
            );
        }
    }

    fclose(fp);
    return true;
}

static bool write_quantization_modulo_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "quantization_modulo_profiles.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(
        fp,
        "split,domain,channel,modulus,dominant_residue,uniformity_score_l1,"
        "residue_0_share,residue_1_share,residue_2_share,residue_3_share,"
        "residue_4_share,residue_5_share,residue_6_share,residue_7_share\n"
    );
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (stats->sample_count[g] == 0 || stats->pixel_count[g] == 0) {
            continue;
        }
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            for (int m = 0; m < MOD_COUNT; ++m) {
                long double profile[8] = {0};
                int modulus = MODULI[m];
                modulo_profile(stats->histogram[g][c], modulus, profile);
                fprintf(
                    fp,
                    "%s,%s,%s,%d,%d,%.10Lf",
                    GROUPS[g].split,
                    GROUPS[g].domain,
                    CHANNELS[c],
                    modulus,
                    dominant_residue(profile, modulus),
                    modulo_uniformity_score(profile, modulus)
                );
                for (int i = 0; i < 8; ++i) {
                    if (i < modulus) {
                        fprintf(fp, ",%.10Lf", profile[i]);
                    } else {
                        fprintf(fp, ",");
                    }
                }
                fprintf(fp, "\n");
            }
        }
    }

    fclose(fp);
    return true;
}

static bool write_quantization_report_md(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "quantization_report.md");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "# Blue Channel Quantization Check\n\n");
    fprintf(fp, "This report compares `R`, `G`, and `B` channel histograms using simple quantization-oriented diagnostics:\n");
    fprintf(fp, "- histogram roughness via first and second differences,\n");
    fprintf(fp, "- even-vs-odd parity bias,\n");
    fprintf(fp, "- residue bias under `mod 2`, `mod 4`, and `mod 8`,\n");
    fprintf(fp, "- and average unique intensity count per image.\n\n");
    fprintf(fp, "Interpretation rule:\n");
    fprintf(fp, "- If `B` consistently has higher roughness and stronger modulo residue bias than `R/G`, that supports the quantization/discretization hypothesis.\n\n");

    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (stats->sample_count[g] == 0 || stats->pixel_count[g] == 0) {
            continue;
        }
        fprintf(fp, "## %s %s\n\n", GROUPS[g].split, GROUPS[g].domain);
        for (int c = 0; c < CHANNEL_COUNT; ++c) {
            long double profile[8] = {0};
            modulo_profile(stats->histogram[g][c], 8, profile);
            fprintf(
                fp,
                "- `%s`: roughness-1=%.10Lf, roughness-2=%.10Lf, parity-bias=%.10Lf, "
                "midrange peak/trough=%.6Lf, mod8 dominant residue=%d, mod8 uniformity-l1=%.10Lf\n",
                CHANNELS[c],
                mean_abs_first_diff(stats->histogram[g][c]),
                mean_abs_second_diff(stats->histogram[g][c]),
                parity_bias(stats->histogram[g][c]),
                peak_to_trough_ratio(stats->histogram[g][c], 40, 100),
                dominant_residue(profile, 8),
                modulo_uniformity_score(profile, 8)
            );
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "Recommendation:\n");
    fprintf(fp, "- Treat the blue-channel histogram as a dataset property if its roughness and modulo bias remain systematically stronger after reruns.\n");
    fprintf(fp, "- Keep normalization explicit and use moderate color augmentation if the blue channel remains more discretized than red/green.\n");

    fclose(fp);
    return true;
}


static bool write_padding_summary_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "padding_summary.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,sample_count,total_pixel_count,valid_pixel_count,black_pixel_count,black_pixel_ratio,images_with_black\n");
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (stats->sample_count[g] == 0 || stats->total_pixel_count[g] == 0) {
            continue;
        }
        long double black_ratio = (long double)stats->black_pixel_count[g] / (long double)stats->total_pixel_count[g];
        fprintf(
            fp,
            "%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.10Lf,%" PRIu64 "\n",
            GROUPS[g].split,
            GROUPS[g].domain,
            stats->sample_count[g],
            stats->total_pixel_count[g],
            stats->pixel_count[g],
            stats->black_pixel_count[g],
            black_ratio,
            stats->images_with_black[g]
        );
    }

    fclose(fp);
    return true;
}

static long parse_long_or_default(const char *text, long default_value, bool allow_zero) {
    if (text == NULL) {
        return default_value;
    }
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return default_value;
    }
    if ((!allow_zero && value <= 0) || (allow_zero && value < 0)) {
        return default_value;
    }
    return value;
}

int main(int argc, char **argv) {
    const char *dataset_root = "data/LoveDA";
    const char *output_dir = "outputs/dataset/full_rgb_mt";
    long thread_count = (long)sysconf(_SC_NPROCESSORS_ONLN);
    long max_images_per_group = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dataset-root") == 0 && i + 1 < argc) {
            dataset_root = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            thread_count = parse_long_or_default(argv[++i], thread_count, false);
        } else if (strcmp(argv[i], "--max-images-per-group") == 0 && i + 1 < argc) {
            max_images_per_group = parse_long_or_default(argv[++i], 0, true);
        } else {
            fprintf(
                stderr,
                "Usage: %s [--dataset-root PATH] [--output-dir PATH] [--threads N] [--max-images-per-group N]\n",
                argv[0]
            );
            return 1;
        }
    }

    if (thread_count <= 0) {
        thread_count = 1;
    }

    if (ensure_dir(output_dir) != 0) {
        fprintf(stderr, "Failed to create output directory %s\n", output_dir);
        return 1;
    }

    FileTask *tasks = NULL;
    size_t task_count = 0;
    size_t task_cap = 0;
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (!collect_group_files(dataset_root, g, &tasks, &task_count, &task_cap, max_images_per_group)) {
            return 1;
        }
    }
    qsort(tasks, task_count, sizeof(FileTask), cmp_tasks);

    TaskQueue queue = {
        .tasks = tasks,
        .task_count = task_count,
        .next_index = 0,
        .mutex = PTHREAD_MUTEX_INITIALIZER,
    };

    pthread_t *threads = (pthread_t *)calloc((size_t)thread_count, sizeof(pthread_t));
    WorkerArgs *args = (WorkerArgs *)calloc((size_t)thread_count, sizeof(WorkerArgs));
    ThreadStats *worker_stats = (ThreadStats *)calloc((size_t)thread_count, sizeof(ThreadStats));
    ThreadStats total_stats;
    if (threads == NULL || args == NULL || worker_stats == NULL) {
        fprintf(stderr, "Failed to allocate worker structures\n");
        return 1;
    }
    init_stats(&total_stats);

    for (long i = 0; i < thread_count; ++i) {
        init_stats(&worker_stats[i]);
        args[i].queue = &queue;
        args[i].stats = &worker_stats[i];
        if (pthread_create(&threads[i], NULL, worker_main, &args[i]) != 0) {
            fprintf(stderr, "Failed to create worker thread %ld\n", i);
            return 1;
        }
    }

    for (long i = 0; i < thread_count; ++i) {
        pthread_join(threads[i], NULL);
        merge_stats(&total_stats, &worker_stats[i]);
    }

    if (!write_global_csv(output_dir, &total_stats)) {
        fprintf(stderr, "Failed to write rgb_global_stats.csv\n");
        return 1;
    }
    if (!write_histogram_csv(output_dir, &total_stats)) {
        fprintf(stderr, "Failed to write rgb_histograms.csv\n");
        return 1;
    }
    if (!write_quantization_summary_csv(output_dir, &total_stats)) {
        fprintf(stderr, "Failed to write quantization_channel_summary.csv\n");
        return 1;
    }
    if (!write_quantization_modulo_csv(output_dir, &total_stats)) {
        fprintf(stderr, "Failed to write quantization_modulo_profiles.csv\n");
        return 1;
    }
    if (!write_quantization_report_md(output_dir, &total_stats)) {
        fprintf(stderr, "Failed to write quantization_report.md\n");
        return 1;
    }
    if (!write_padding_summary_csv(output_dir, &total_stats)) {
        fprintf(stderr, "Failed to write padding_summary.csv\n");
        return 1;
    }

    printf("Processed %zu images with %ld threads\n", task_count, thread_count);
    printf("Saved outputs to %s\n", output_dir);

    for (size_t i = 0; i < task_count; ++i) {
        free(tasks[i].path);
    }
    free(tasks);
    free(threads);
    free(args);
    free(worker_stats);
    pthread_mutex_destroy(&queue.mutex);
    return 0;
}
