#define _POSIX_C_SOURCE 200809L
#define STB_IMAGE_IMPLEMENTATION

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
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

#define GROUP_COUNT 4
#define CLASS_COUNT 8

typedef struct {
    const char *split;
    const char *domain;
    const char *rel_dir;
} GroupInfo;

static const GroupInfo GROUPS[GROUP_COUNT] = {
    {"Train", "Urban", "Train/Urban/masks_png"},
    {"Train", "Rural", "Train/Rural/masks_png"},
    {"Val", "Urban", "Val/Urban/masks_png"},
    {"Val", "Rural", "Val/Rural/masks_png"},
};

static const char *CLASS_NAMES[CLASS_COUNT] = {
    "no-data",
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
};

typedef struct {
    char *path;
    char *sample_id;
    int group_id;
} FileTask;

typedef struct {
    char *sample_id;
    int group_id;
    uint64_t total_pixels;
    uint64_t class_count[CLASS_COUNT];
    double boundary_density;
} SampleRow;

typedef struct {
    uint64_t sample_count[GROUP_COUNT];
    uint64_t pixel_count[GROUP_COUNT];
    uint64_t class_count[GROUP_COUNT][CLASS_COUNT];
    uint64_t cooccurrence[GROUP_COUNT][CLASS_COUNT][CLASS_COUNT];
    uint64_t adjacency[GROUP_COUNT][CLASS_COUNT][CLASS_COUNT];
    SampleRow *rows;
    size_t row_count;
    size_t row_cap;
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

static int cmp_tasks(const void *lhs, const void *rhs) {
    const FileTask *a = (const FileTask *)lhs;
    const FileTask *b = (const FileTask *)rhs;
    int group_cmp = a->group_id - b->group_id;
    if (group_cmp != 0) {
        return group_cmp;
    }
    return strcmp(a->path, b->path);
}

static int cmp_rows(const void *lhs, const void *rhs) {
    const SampleRow *a = (const SampleRow *)lhs;
    const SampleRow *b = (const SampleRow *)rhs;
    int group_cmp = a->group_id - b->group_id;
    if (group_cmp != 0) {
        return group_cmp;
    }
    return strcmp(a->sample_id, b->sample_id);
}

static bool has_png_suffix(const char *name) {
    size_t len = strlen(name);
    return len >= 4 && strcmp(name + len - 4, ".png") == 0;
}

static char *dup_string(const char *src) {
    size_t len = strlen(src) + 1;
    char *dst = (char *)malloc(len);
    if (dst != NULL) {
        memcpy(dst, src, len);
    }
    return dst;
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

static char *sample_id_from_name(const char *name) {
    size_t len = strlen(name);
    size_t stem_len = len >= 4 ? len - 4 : len;
    char *buf = (char *)malloc(stem_len + 1);
    if (buf == NULL) {
        return NULL;
    }
    memcpy(buf, name, stem_len);
    buf[stem_len] = '\0';
    return buf;
}

static int ensure_dir(const char *path) {
    char *tmp = dup_string(path);
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
    size_t *task_cap
) {
    char *dir_path = join_path(dataset_root, GROUPS[group_id].rel_dir);
    if (dir_path == NULL) {
        return false;
    }

    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        free(dir_path);
        return true;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.' || !has_png_suffix(entry->d_name)) {
            continue;
        }
        if (*task_count == *task_cap) {
            size_t new_cap = *task_cap == 0 ? 1024 : *task_cap * 2;
            FileTask *new_tasks = (FileTask *)realloc(*tasks, new_cap * sizeof(FileTask));
            if (new_tasks == NULL) {
                closedir(dir);
                free(dir_path);
                return false;
            }
            *tasks = new_tasks;
            *task_cap = new_cap;
        }

        char *file_path = join_path(dir_path, entry->d_name);
        char *sample_id = sample_id_from_name(entry->d_name);
        if (file_path == NULL || sample_id == NULL) {
            free(file_path);
            free(sample_id);
            closedir(dir);
            free(dir_path);
            return false;
        }

        (*tasks)[*task_count].path = file_path;
        (*tasks)[*task_count].sample_id = sample_id;
        (*tasks)[*task_count].group_id = group_id;
        *task_count += 1;
    }

    closedir(dir);
    free(dir_path);
    return true;
}

static void init_thread_stats(ThreadStats *stats) {
    memset(stats, 0, sizeof(*stats));
}

static bool append_row(ThreadStats *stats, const SampleRow *row) {
    if (stats->row_count == stats->row_cap) {
        size_t new_cap = stats->row_cap == 0 ? 512 : stats->row_cap * 2;
        SampleRow *new_rows = (SampleRow *)realloc(stats->rows, new_cap * sizeof(SampleRow));
        if (new_rows == NULL) {
            return false;
        }
        stats->rows = new_rows;
        stats->row_cap = new_cap;
    }
    stats->rows[stats->row_count++] = *row;
    return true;
}

static void process_mask(ThreadStats *stats, const FileTask *task, const unsigned char *pixels, int width, int height) {
    SampleRow row;
    memset(&row, 0, sizeof(row));
    row.sample_id = dup_string(task->sample_id);
    row.group_id = task->group_id;
    row.total_pixels = (uint64_t)width * (uint64_t)height;
    if (row.sample_id == NULL) {
        return;
    }

    stats->sample_count[task->group_id] += 1;
    stats->pixel_count[task->group_id] += row.total_pixels;

    uint64_t total = row.total_pixels;
    for (uint64_t i = 0; i < total; ++i) {
        unsigned int value = pixels[i];
        if (value < CLASS_COUNT) {
            row.class_count[value] += 1;
            stats->class_count[task->group_id][value] += 1;
        }
    }

    bool present[CLASS_COUNT] = {false};
    for (int c = 0; c < CLASS_COUNT; ++c) {
        present[c] = row.class_count[c] > 0;
    }
    for (int i = 0; i < CLASS_COUNT; ++i) {
        if (!present[i]) {
            continue;
        }
        for (int j = 0; j < CLASS_COUNT; ++j) {
            if (present[j]) {
                stats->cooccurrence[task->group_id][i][j] += 1;
            }
        }
    }

    uint64_t boundary_transitions = 0;
    uint64_t adjacency_total = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x + 1 < width; ++x) {
            uint8_t left = pixels[y * width + x];
            uint8_t right = pixels[y * width + x + 1];
            adjacency_total += 1;
            if (left != right && left < CLASS_COUNT && right < CLASS_COUNT) {
                stats->adjacency[task->group_id][left][right] += 1;
                stats->adjacency[task->group_id][right][left] += 1;
                boundary_transitions += 1;
            }
        }
    }
    for (int y = 0; y + 1 < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t top = pixels[y * width + x];
            uint8_t bottom = pixels[(y + 1) * width + x];
            adjacency_total += 1;
            if (top != bottom && top < CLASS_COUNT && bottom < CLASS_COUNT) {
                stats->adjacency[task->group_id][top][bottom] += 1;
                stats->adjacency[task->group_id][bottom][top] += 1;
                boundary_transitions += 1;
            }
        }
    }
    row.boundary_density = adjacency_total > 0
        ? (double)boundary_transitions / (double)adjacency_total
        : 0.0;

    if (!append_row(stats, &row)) {
        free(row.sample_id);
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
        unsigned char *pixels = stbi_load(task.path, &width, &height, &channels, 1);
        if (pixels == NULL) {
            fprintf(stderr, "Failed to load %s: %s\n", task.path, stbi_failure_reason());
            continue;
        }
        process_mask(worker->stats, &task, pixels, width, height);
        stbi_image_free(pixels);
    }
    return NULL;
}

static bool merge_rows(ThreadStats *dst, const ThreadStats *src) {
    for (size_t i = 0; i < src->row_count; ++i) {
        if (!append_row(dst, &src->rows[i])) {
            return false;
        }
    }
    return true;
}

static bool merge_stats(ThreadStats *dst, const ThreadStats *src) {
    for (int g = 0; g < GROUP_COUNT; ++g) {
        dst->sample_count[g] += src->sample_count[g];
        dst->pixel_count[g] += src->pixel_count[g];
        for (int c = 0; c < CLASS_COUNT; ++c) {
            dst->class_count[g][c] += src->class_count[g][c];
            for (int j = 0; j < CLASS_COUNT; ++j) {
                dst->cooccurrence[g][c][j] += src->cooccurrence[g][c][j];
                dst->adjacency[g][c][j] += src->adjacency[g][c][j];
            }
        }
    }
    return merge_rows(dst, src);
}

static bool write_group_summary_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "mask_group_summary.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,sample_count,pixel_count\n");
    for (int g = 0; g < GROUP_COUNT; ++g) {
        fprintf(
            fp,
            "%s,%s,%" PRIu64 ",%" PRIu64 "\n",
            GROUPS[g].split,
            GROUPS[g].domain,
            stats->sample_count[g],
            stats->pixel_count[g]
        );
    }
    fclose(fp);
    return true;
}

static bool write_group_distribution_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "mask_group_class_distribution.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,class_id,class_name,pixel_count,pixel_ratio\n");
    for (int g = 0; g < GROUP_COUNT; ++g) {
        long double total = (long double)stats->pixel_count[g];
        for (int c = 0; c < CLASS_COUNT; ++c) {
            long double ratio = total > 0.0L ? (long double)stats->class_count[g][c] / total : 0.0L;
            fprintf(
                fp,
                "%s,%s,%d,%s,%" PRIu64 ",%.8Lf\n",
                GROUPS[g].split,
                GROUPS[g].domain,
                c,
                CLASS_NAMES[c],
                stats->class_count[g][c],
                ratio
            );
        }
    }
    fclose(fp);
    return true;
}

static bool write_sample_counts_csv(const char *output_dir, ThreadStats *stats) {
    qsort(stats->rows, stats->row_count, sizeof(SampleRow), cmp_rows);

    char *path = join_path(output_dir, "mask_sample_class_counts.csv");
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
        "split,domain,sample_id,total_pixels,present_class_count,dominant_class_id,dominant_class_name,dominant_ratio,boundary_density,"
        "count_0,count_1,count_2,count_3,count_4,count_5,count_6,count_7,"
        "ratio_0,ratio_1,ratio_2,ratio_3,ratio_4,ratio_5,ratio_6,ratio_7\n"
    );

    for (size_t i = 0; i < stats->row_count; ++i) {
        const SampleRow *row = &stats->rows[i];
        int dominant_class = 0;
        uint64_t dominant_count = row->class_count[0];
        int present_count = 0;
        for (int c = 0; c < CLASS_COUNT; ++c) {
            if (row->class_count[c] > 0) {
                present_count += 1;
            }
            if (row->class_count[c] > dominant_count) {
                dominant_count = row->class_count[c];
                dominant_class = c;
            }
        }
        long double dominant_ratio = row->total_pixels > 0
            ? (long double)dominant_count / (long double)row->total_pixels
            : 0.0L;

        fprintf(
            fp,
            "%s,%s,%s,%" PRIu64 ",%d,%d,%s,%.8Lf,%.8f",
            GROUPS[row->group_id].split,
            GROUPS[row->group_id].domain,
            row->sample_id,
            row->total_pixels,
            present_count,
            dominant_class,
            CLASS_NAMES[dominant_class],
            dominant_ratio,
            row->boundary_density
        );
        for (int c = 0; c < CLASS_COUNT; ++c) {
            fprintf(fp, ",%" PRIu64, row->class_count[c]);
        }
        for (int c = 0; c < CLASS_COUNT; ++c) {
            long double ratio = row->total_pixels > 0
                ? (long double)row->class_count[c] / (long double)row->total_pixels
                : 0.0L;
            fprintf(fp, ",%.8Lf", ratio);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return true;
}

static bool write_cooccurrence_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "class_cooccurrence.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,class_id_i,class_name_i,class_id_j,class_name_j,image_count\n");
    for (int g = 0; g < GROUP_COUNT; ++g) {
        for (int i = 0; i < CLASS_COUNT; ++i) {
            for (int j = 0; j < CLASS_COUNT; ++j) {
                fprintf(
                    fp,
                    "%s,%s,%d,%s,%d,%s,%" PRIu64 "\n",
                    GROUPS[g].split,
                    GROUPS[g].domain,
                    i,
                    CLASS_NAMES[i],
                    j,
                    CLASS_NAMES[j],
                    stats->cooccurrence[g][i][j]
                );
            }
        }
    }
    fclose(fp);
    return true;
}

static bool write_adjacency_csv(const char *output_dir, const ThreadStats *stats) {
    char *path = join_path(output_dir, "class_adjacency.csv");
    if (path == NULL) {
        return false;
    }
    FILE *fp = fopen(path, "w");
    free(path);
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,class_id_i,class_name_i,class_id_j,class_name_j,boundary_touch_count\n");
    for (int g = 0; g < GROUP_COUNT; ++g) {
        for (int i = 0; i < CLASS_COUNT; ++i) {
            for (int j = 0; j < CLASS_COUNT; ++j) {
                fprintf(
                    fp,
                    "%s,%s,%d,%s,%d,%s,%" PRIu64 "\n",
                    GROUPS[g].split,
                    GROUPS[g].domain,
                    i,
                    CLASS_NAMES[i],
                    j,
                    CLASS_NAMES[j],
                    stats->adjacency[g][i][j]
                );
            }
        }
    }
    fclose(fp);
    return true;
}

static long parse_long_or_default(const char *text, long default_value) {
    if (text == NULL) {
        return default_value;
    }
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0) {
        return default_value;
    }
    return value;
}

int main(int argc, char **argv) {
    const char *dataset_root = "data/LoveDA";
    const char *output_dir = "outputs/dataset/mask_mt";
    long thread_count = (long)sysconf(_SC_NPROCESSORS_ONLN);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dataset-root") == 0 && i + 1 < argc) {
            dataset_root = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            thread_count = parse_long_or_default(argv[++i], thread_count);
        } else {
            fprintf(stderr, "Usage: %s [--dataset-root PATH] [--output-dir PATH] [--threads N]\n", argv[0]);
            return 1;
        }
    }

    if (ensure_dir(output_dir) != 0) {
        fprintf(stderr, "Failed to create output directory %s\n", output_dir);
        return 1;
    }

    FileTask *tasks = NULL;
    size_t task_count = 0;
    size_t task_cap = 0;
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (!collect_group_files(dataset_root, g, &tasks, &task_count, &task_cap)) {
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
        return 1;
    }
    init_thread_stats(&total_stats);

    for (long i = 0; i < thread_count; ++i) {
        init_thread_stats(&worker_stats[i]);
        args[i].queue = &queue;
        args[i].stats = &worker_stats[i];
        if (pthread_create(&threads[i], NULL, worker_main, &args[i]) != 0) {
            return 1;
        }
    }

    for (long i = 0; i < thread_count; ++i) {
        pthread_join(threads[i], NULL);
        if (!merge_stats(&total_stats, &worker_stats[i])) {
            return 1;
        }
    }

    if (!write_group_summary_csv(output_dir, &total_stats) ||
        !write_group_distribution_csv(output_dir, &total_stats) ||
        !write_sample_counts_csv(output_dir, &total_stats) ||
        !write_cooccurrence_csv(output_dir, &total_stats) ||
        !write_adjacency_csv(output_dir, &total_stats)) {
        return 1;
    }

    printf("Processed %zu masks with %ld threads\n", task_count, thread_count);
    printf("Saved outputs to %s\n", output_dir);

    for (size_t i = 0; i < task_count; ++i) {
        free(tasks[i].path);
        free(tasks[i].sample_id);
    }
    for (size_t i = 0; i < total_stats.row_count; ++i) {
        free(total_stats.rows[i].sample_id);
    }
    for (long i = 0; i < thread_count; ++i) {
        free(worker_stats[i].rows);
    }
    free(total_stats.rows);
    free(tasks);
    free(threads);
    free(args);
    free(worker_stats);
    pthread_mutex_destroy(&queue.mutex);
    return 0;
}
