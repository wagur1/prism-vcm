/* Compile against the exact pinned x265 header/library; never mirror structs in ctypes. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x265.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    const x265_api *api;
    x265_param *param;
    x265_encoder *encoder;
    x265_picture *input;
    x265_picture *output;
    unsigned char *packet;
    size_t capacity;
} prism_encoder;

EXPORT int prism_abi(void) { return X265_BUILD; }
EXPORT const char *prism_version(void) { return x265_version_str; }
EXPORT size_t prism_picture_size(void) { return sizeof(x265_picture); }

EXPORT void prism_close(prism_encoder *ctx) {
    if (!ctx) return;
    if (ctx->encoder) ctx->api->encoder_close(ctx->encoder);
    if (ctx->input) ctx->api->picture_free(ctx->input);
    if (ctx->output) ctx->api->picture_free(ctx->output);
    if (ctx->param) ctx->api->param_free(ctx->param);
    free(ctx->packet);
    free(ctx);
}

EXPORT prism_encoder *prism_open(int width, int height, const char *preset,
                                const char *options) {
    prism_encoder *ctx = (prism_encoder *)calloc(1, sizeof(*ctx));
    char *copy, *line;
    if (!ctx) return NULL;
    ctx->api = x265_api_get(8);
    if (!ctx->api || ctx->api->sizeof_picture != sizeof(x265_picture) ||
        ctx->api->sizeof_param != sizeof(x265_param)) goto fail;
    ctx->param = ctx->api->param_alloc();
    if (!ctx->param || ctx->api->param_default_preset(ctx->param, preset, NULL)) goto fail;
    ctx->param->sourceWidth = width;
    ctx->param->sourceHeight = height;
    ctx->param->internalCsp = X265_CSP_I420;
    ctx->param->internalBitDepth = 8;
    copy = (char *)malloc(strlen(options) + 1);
    if (!copy) goto fail;
    strcpy(copy, options);
    line = strtok(copy, "\n");
    while (line) {
        char *equals = strchr(line, '=');
        if (!equals) { free(copy); goto fail; }
        *equals = 0;
        if (ctx->api->param_parse(ctx->param, line, equals + 1)) { free(copy); goto fail; }
        line = strtok(NULL, "\n");
    }
    free(copy);
    ctx->encoder = ctx->api->encoder_open(ctx->param);
    ctx->input = ctx->api->picture_alloc();
    ctx->output = ctx->api->picture_alloc();
    if (!ctx->encoder || !ctx->input || !ctx->output) goto fail;
    ctx->api->picture_init(ctx->param, ctx->input);
    ctx->api->picture_init(ctx->param, ctx->output);
    return ctx;
fail:
    prism_close(ctx);
    return NULL;
}

/* Inputs remain owned by Python until output PTS proves the frame was consumed. */
EXPORT int prism_encode(prism_encoder *ctx, uint8_t *yuv, float *offsets,
                        int width, int height, int64_t pts,
                        uint8_t **packet, size_t *length, int64_t *out_pts) {
    x265_nal *nals = NULL;
    uint32_t count = 0, i;
    size_t total = 0, position = 0;
    int result;
    if (yuv) {
        ctx->input->planes[0] = yuv;
        ctx->input->planes[1] = yuv + width * height;
        ctx->input->planes[2] = yuv + width * height * 5 / 4;
        ctx->input->stride[0] = width;
        ctx->input->stride[1] = ctx->input->stride[2] = width / 2;
        ctx->input->pts = pts;
        ctx->input->quantOffsets = offsets;
    }
    result = ctx->api->encoder_encode(ctx->encoder, &nals, &count,
                                     yuv ? ctx->input : NULL, ctx->output);
    if (result < 0) return result;
    for (i = 0; i < count; ++i) total += nals[i].sizeBytes;
    if (total > ctx->capacity) {
        unsigned char *grown = (unsigned char *)realloc(ctx->packet, total);
        if (!grown) return -1;
        ctx->packet = grown;
        ctx->capacity = total;
    }
    for (i = 0; i < count; ++i) {
        memcpy(ctx->packet + position, nals[i].payload, nals[i].sizeBytes);
        position += nals[i].sizeBytes;
    }
    *packet = ctx->packet;
    *length = total;
    *out_pts = result ? ctx->output->pts : -1;
    return result;
}
