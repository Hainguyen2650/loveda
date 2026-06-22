#define _POSIX_C_SOURCE 200809L
#define STB_IMAGE_IMPLEMENTATION

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
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

typedef struct {
    char *path;
    char *sample_id;
    int group_id;
} FileTask;

typedef struct {
    char *path;
    char *sample_id;
    int group_id;
    char *mask_path;
    uint64_t black_pixels;
    uint64_t total_pixels;
    long double black_ratio;
    bool flagged;
    bool image_deleted;
    bool mask_deleted;
} ResultRow;

typedef struct {
    FileTask *tasks;
    size_t task_count;
    size_t next_index;
    pthread_mutex_t mutex;
} TaskQueue;

typedef struct {
    TaskQueue *queue;
    ResultRow *results;
    long double threshold;
} WorkerArgs;

static bool has_png_suffix(const char *name) {
    size_t len = strlen(name);
    return len >= 4 && strcmp(name + len - 4, ".png") == 0;
}

static char *dup_string(const char *src) {
    size_t len = strlen(src);
    char *dst = (char *)malloc(len + 1);
    if (dst == NULL) {
        return NULL;
    }
    memcpy(dst, src, len + 1);
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

static bool file_exists(const char *path) {
    return access(path, F_OK) == 0;
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

static int cmp_tasks(const void *lhs, const void *rhs) {
    const FileTask *a = (const FileTask *)lhs;
    const FileTask *b = (const FileTask *)rhs;
    int group_cmp = a->group_id - b->group_id;
    if (group_cmp != 0) {
        return group_cmp;
    }
    return strcmp(a->path, b->path);
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
        fprintf(stderr, "Failed to allocate directory path\n");
        return false;
    }

    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        free(dir_path);
        return true;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (!has_png_suffix(entry->d_name)) {
            continue;
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

        char *sample_id = dup_string(entry->d_name);
        if (sample_id == NULL) {
            free(file_path);
            closedir(dir);
            free(dir_path);
            return false;
        }
        char *dot = strrchr(sample_id, '.');
        if (dot != NULL) {
            *dot = '\0';
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

static void process_image(const FileTask *task, long double threshold, ResultRow *result) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char *pixels = stbi_load(task->path, &width, &height, &channels, CHANNEL_COUNT);
    if (pixels == NULL) {
        fprintf(stderr, "Failed to load %s: %s\n", task->path, stbi_failure_reason());
        result->path = dup_string(task->path);
        result->sample_id = dup_string(task->sample_id);
        result->group_id = task->group_id;
        result->mask_path = NULL;
        result->black_pixels = 0;
        result->total_pixels = 0;
        result->black_ratio = 0.0L;
        result->flagged = false;
        result->image_deleted = false;
        result->mask_deleted = false;
        return;
    }

    uint64_t total_pixels = (uint64_t)width * (uint64_t)height;
    uint64_t black_pixels = 0;
    const unsigned char *ptr = pixels;
    for (uint64_t i = 0; i < total_pixels; ++i) {
        if (ptr[0] == 0 && ptr[1] == 0 && ptr[2] == 0) {
            black_pixels += 1;
        }
        ptr += CHANNEL_COUNT;
    }
    stbi_image_free(pixels);

    long double black_ratio = total_pixels == 0 ? 0.0L : (long double)black_pixels / (long double)total_pixels;

    result->path = dup_string(task->path);
    result->sample_id = dup_string(task->sample_id);
    result->group_id = task->group_id;
    result->mask_path = NULL;
    result->black_pixels = black_pixels;
    result->total_pixels = total_pixels;
    result->black_ratio = black_ratio;
    result->flagged = black_ratio > threshold;
    result->image_deleted = false;
    result->mask_deleted = false;
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

        process_image(&task, worker->threshold, &worker->results[index]);
    }

    return NULL;
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

static long double parse_threshold_or_default(const char *text, long double default_value) {
    if (text == NULL) {
        return default_value;
    }
    char *end = NULL;
    long double value = strtold(text, &end);
    if (end == text || *end != '\0' || value < 0.0L || value > 1.0L) {
        return default_value;
    }
    return value;
}

static long double parse_percent_threshold_or_default(const char *text, long double default_value) {
    if (text == NULL) {
        return default_value;
    }
    char *end = NULL;
    long double value = strtold(text, &end);
    if (end == text || *end != '\0' || value < 0.0L || value > 100.0L) {
        return default_value;
    }
    return value / 100.0L;
}

static bool write_flagged_csv(const char *output_csv, const ResultRow *results, size_t result_count) {
    FILE *fp = fopen(output_csv, "w");
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "split,domain,sample_id,image_path,mask_path,black_pixels,total_pixels,black_ratio,image_deleted,mask_deleted\n");
    for (size_t i = 0; i < result_count; ++i) {
        if (!results[i].flagged) {
            continue;
        }
        int group_id = results[i].group_id;
        fprintf(
            fp,
            "%s,%s,%s,%s,%s,%" PRIu64 ",%" PRIu64 ",%.10Lf,%s,%s\n",
            GROUPS[group_id].split,
            GROUPS[group_id].domain,
            results[i].sample_id,
            results[i].path,
            results[i].mask_path != NULL ? results[i].mask_path : "",
            results[i].black_pixels,
            results[i].total_pixels,
            results[i].black_ratio,
            results[i].image_deleted ? "true" : "false",
            results[i].mask_deleted ? "true" : "false"
        );
    }

    fclose(fp);
    return true;
}

static void attach_mask_paths(const char *dataset_root, ResultRow *results, size_t result_count) {
    for (size_t i = 0; i < result_count; ++i) {
        if (!results[i].flagged) {
            continue;
        }
        int group_id = results[i].group_id;
        char rel_mask_dir[256];
        snprintf(
            rel_mask_dir,
            sizeof(rel_mask_dir),
            "%s/%s/masks_png/%s.png",
            GROUPS[group_id].split,
            GROUPS[group_id].domain,
            results[i].sample_id
        );
        char *mask_path = join_path(dataset_root, rel_mask_dir);
        if (mask_path != NULL && file_exists(mask_path)) {
            results[i].mask_path = mask_path;
        } else {
            free(mask_path);
            results[i].mask_path = NULL;
        }
    }
}

static void delete_flagged_files(ResultRow *results, size_t result_count) {
    for (size_t i = 0; i < result_count; ++i) {
        if (!results[i].flagged) {
            continue;
        }
        if (results[i].path != NULL && file_exists(results[i].path) && remove(results[i].path) == 0) {
            results[i].image_deleted = true;
        }
        if (results[i].mask_path != NULL && file_exists(results[i].mask_path) && remove(results[i].mask_path) == 0) {
            results[i].mask_deleted = true;
        }
    }
}

int main(int argc, char **argv) {
    const char *dataset_root = "data/LoveDA";
    const char *output_csv = "outputs/dataset/large_padding_images.csv";
    long double threshold = 0.10L;
    long thread_count = (long)sysconf(_SC_NPROCESSORS_ONLN);
    bool delete_flagged = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dataset-root") == 0 && i + 1 < argc) {
            dataset_root = argv[++i];
        } else if (strcmp(argv[i], "--output-csv") == 0 && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            threshold = parse_threshold_or_default(argv[++i], threshold);
        } else if (strcmp(argv[i], "--threshold-percent") == 0 && i + 1 < argc) {
            threshold = parse_percent_threshold_or_default(argv[++i], threshold);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            thread_count = parse_long_or_default(argv[++i], thread_count);
        } else if (strcmp(argv[i], "--delete-flagged") == 0) {
            delete_flagged = true;
        } else {
            fprintf(
                stderr,
                "Usage: %s [--dataset-root PATH] [--output-csv PATH] [--threshold RATIO] [--threshold-percent PERCENT] [--threads N] [--delete-flagged]\n",
                argv[0]
            );
            return 1;
        }
    }

    if (thread_count <= 0) {
        thread_count = 1;
    }

    char *output_dir = dup_string(output_csv);
    if (output_dir == NULL) {
        fprintf(stderr, "Failed to allocate output dir buffer\n");
        return 1;
    }
    char *slash = strrchr(output_dir, '/');
    if (slash != NULL) {
        *slash = '\0';
        if (ensure_dir(output_dir) != 0) {
            fprintf(stderr, "Failed to create output directory %s\n", output_dir);
            free(output_dir);
            return 1;
        }
    }
    free(output_dir);

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

    ResultRow *results = (ResultRow *)calloc(task_count, sizeof(ResultRow));
    pthread_t *threads = (pthread_t *)calloc((size_t)thread_count, sizeof(pthread_t));
    WorkerArgs *args = (WorkerArgs *)calloc((size_t)thread_count, sizeof(WorkerArgs));
    if (results == NULL || threads == NULL || args == NULL) {
        fprintf(stderr, "Failed to allocate worker structures\n");
        return 1;
    }

    for (long i = 0; i < thread_count; ++i) {
        args[i].queue = &queue;
        args[i].results = results;
        args[i].threshold = threshold;
        if (pthread_create(&threads[i], NULL, worker_main, &args[i]) != 0) {
            fprintf(stderr, "Failed to create worker thread %ld\n", i);
            return 1;
        }
    }

    for (long i = 0; i < thread_count; ++i) {
        pthread_join(threads[i], NULL);
    }

    attach_mask_paths(dataset_root, results, task_count);
    if (delete_flagged) {
        delete_flagged_files(results, task_count);
    }

    if (!write_flagged_csv(output_csv, results, task_count)) {
        fprintf(stderr, "Failed to write %s\n", output_csv);
        return 1;
    }

    uint64_t total_flagged = 0;
    uint64_t group_counts[GROUP_COUNT];
    memset(group_counts, 0, sizeof(group_counts));
    for (size_t i = 0; i < task_count; ++i) {
        if (results[i].flagged) {
            total_flagged += 1;
            group_counts[results[i].group_id] += 1;
        }
    }

    printf("Threshold: %.4Lf\n", threshold);
    printf("Flagged images: %" PRIu64 "\n", total_flagged);
    for (int g = 0; g < GROUP_COUNT; ++g) {
        if (group_counts[g] == 0) {
            continue;
        }
        printf("%s %s: %" PRIu64 "\n", GROUPS[g].split, GROUPS[g].domain, group_counts[g]);
    }
    if (delete_flagged) {
        uint64_t image_deleted_count = 0;
        uint64_t mask_deleted_count = 0;
        for (size_t i = 0; i < task_count; ++i) {
            image_deleted_count += results[i].image_deleted ? 1 : 0;
            mask_deleted_count += results[i].mask_deleted ? 1 : 0;
        }
        printf("Deleted images: %" PRIu64 "\n", image_deleted_count);
        printf("Deleted masks: %" PRIu64 "\n", mask_deleted_count);
    }
    printf("Saved detailed rows to %s\n", output_csv);

    for (size_t i = 0; i < task_count; ++i) {
        free(tasks[i].path);
        free(tasks[i].sample_id);
        free(results[i].path);
        free(results[i].sample_id);
        free(results[i].mask_path);
    }
    free(tasks);
    free(results);
    free(threads);
    free(args);
    pthread_mutex_destroy(&queue.mutex);
    return 0;
}
