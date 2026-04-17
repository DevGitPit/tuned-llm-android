#include <jni.h>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <android/log.h>
#include "llama.h"

#ifdef GGML_USE_OPENBLAS
extern "C" {
    void openblas_set_num_threads(int num_threads);
}
#endif

#define TAG "LLM_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::atomic<bool> stop_flag{false};
    std::vector<llama_token> previous_tokens;
} g_state;

static JavaVM* g_vm = nullptr;
static std::mutex g_inference_mutex;
static std::atomic<bool> g_is_generating{false};

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_vm = vm;
    
#ifdef GGML_USE_OPENBLAS
    LOGI("Setting OpenBLAS threads to 1 for better stability");
    openblas_set_num_threads(1);
#endif

    return JNI_VERSION_1_6;
}

// Helper to safely create a JNI string from potentially invalid UTF-8
static jstring safe_new_string_utf(JNIEnv* env, const char* str) {
    if (!str) return nullptr;
    jstring result = env->NewStringUTF(str);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        LOGW("Invalid UTF-8 detected in piece, returning placeholder");
        return env->NewStringUTF("");
    }
    if (!result) {
        LOGW("Failed to create JNI string, returning placeholder");
        return env->NewStringUTF("");
    }
    return result;
}

// Helper to add a token to a batch
static void batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < (size_t)seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

static void inference_thread(jobject callback_global, std::string prompt_str, std::vector<std::string> stop_strings) {
    g_is_generating = true;
    JNIEnv* env;
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
        LOGE("Failed to attach current thread");
        g_is_generating = false;
        return;
    }

    LOGI("Inference thread starting");

    jclass callback_class = env->GetObjectClass(callback_global);
    jmethodID onToken_id = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
    jmethodID onComplete_id = env->GetMethodID(callback_class, "onComplete", "()V");
    jmethodID onError_id = env->GetMethodID(callback_class, "onError", "(Ljava/lang/String;)V");

    const struct llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    llama_batch batch = llama_batch_init(llama_n_batch(g_state.ctx), 0, 1);
    llama_sampler* smpl = nullptr;

    try {
        if (!g_state.model || !g_state.ctx || !vocab) {
            throw std::runtime_error("Model, context or vocab not initialized");
        }

        // Tokenize prompt
        std::vector<llama_token> prompt_tokens(prompt_str.size() + 4);
        int n_prompt_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
        if (n_prompt_tokens < 0) {
            throw std::runtime_error("Tokenization failed");
        }
        prompt_tokens.resize(n_prompt_tokens);
        LOGI("Prompt tokenized into %zu tokens", prompt_tokens.size());

        // Cache reuse check
        size_t n_past = 0;
        while (n_past < g_state.previous_tokens.size() && 
               n_past < prompt_tokens.size() && 
               g_state.previous_tokens[n_past] == prompt_tokens[n_past]) {
            n_past++;
        }

        llama_memory_t mem = llama_get_memory(g_state.ctx);
        if (n_past < g_state.previous_tokens.size()) {
            LOGI("Partial cache hit: n_past = %zu. Clearing stale tokens from KV cache.", n_past);
            if (n_past == 0) {
                if (mem) llama_memory_clear(mem, true);
            } else {
                if (mem) llama_memory_seq_rm(mem, -1, (llama_pos)n_past, -1);
            }
        } else {
            LOGI("Cache hit: n_past = %zu. No clearing needed.", n_past);
        }

        if (prompt_tokens.size() > (size_t)llama_n_ctx(g_state.ctx)) {
            LOGE("Prompt too long (%zu) for context window (%d)", prompt_tokens.size(), llama_n_ctx(g_state.ctx));
            throw std::runtime_error("Prompt exceeds context window limit");
        }

        int n_batch = llama_n_batch(g_state.ctx);
        std::vector<llama_token> session_tokens = prompt_tokens;

        // Prompt Processing (KV Cache Fill)
        bool prompt_interrupted = false;
        for (size_t i = n_past; i < prompt_tokens.size(); i += n_batch) {
            if (g_state.stop_flag.load()) {
                LOGI("Prompt processing interrupted");
                prompt_interrupted = true;
                break;
            }

            int n_eval = (int)std::min((size_t)n_batch, prompt_tokens.size() - i);
            batch.n_tokens = 0;
            for (int j = 0; j < n_eval; ++j) {
                batch_add(batch, prompt_tokens[i + j], (llama_pos)(i + j), {0}, j == n_eval - 1);
            }

            int decode_res = llama_decode(g_state.ctx, batch);
            if (decode_res != 0) {
                LOGE("llama_decode failed during prompt processing with code: %d at i: %zu", decode_res, i);
                throw std::runtime_error("Prompt processing failed (decode code -1)");
            }
        }
        
        if (prompt_interrupted) {
            g_state.previous_tokens.clear();
            if (mem) llama_memory_clear(mem, true);
            throw std::runtime_error("Generation stopped during prompt processing");
        }

        n_past = prompt_tokens.size();

        // Generation Loop
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        LOGI("Starting generation loop at n_past: %zu", n_past);

        while (n_past < (size_t)llama_n_ctx(g_state.ctx)) {
            if (g_state.stop_flag.load()) {
                LOGI("Generation interrupted");
                break;
            }

            llama_token new_token_id = llama_sampler_sample(smpl, g_state.ctx, -1);
            session_tokens.push_back(new_token_id);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                LOGI("EOG token detected: %d", new_token_id);
                break;
            }

            char buf[512];
            int n_piece = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            std::string piece;
            if (n_piece < 0) {
                std::vector<char> dynamic_buf(-n_piece);
                int actual_piece = llama_token_to_piece(vocab, new_token_id, dynamic_buf.data(), dynamic_buf.size(), 0, true);
                piece = std::string(dynamic_buf.data(), actual_piece);
            } else {
                piece = std::string(buf, n_piece);
            }

            bool stop_found = false;
            for (const auto& stop_str : stop_strings) {
                if (!stop_str.empty() && piece.find(stop_str) != std::string::npos) {
                    LOGI("Stop string detected: %s", stop_str.c_str());
                    stop_found = true;
                    break;
                }
            }
            if (stop_found) break;

            if (!piece.empty()) {
                jstring jtoken = safe_new_string_utf(env, piece.c_str());
                if (jtoken) {
                    env->CallVoidMethod(callback_global, onToken_id, jtoken);
                    env->DeleteLocalRef(jtoken);
                }
            }

            batch.n_tokens = 0;
            batch_add(batch, new_token_id, (llama_pos)n_past, {0}, true);
            int decode_res = llama_decode(g_state.ctx, batch);
            if (decode_res != 0) {
                LOGE("llama_decode failed during generation with code: %d at n_past: %zu", decode_res, n_past);
                break; 
            }
            n_past++;
        }

        if (n_past >= (size_t)llama_n_ctx(g_state.ctx)) {
            LOGW("Context window reached: %zu", n_past);
        }

        g_state.previous_tokens = session_tokens;

        if (!g_state.stop_flag.load()) {
            env->CallVoidMethod(callback_global, onComplete_id);
        }

    } catch (const std::exception& e) {
        g_state.previous_tokens.clear();
        LOGE("Inference error: %s", e.what());
        jstring jmsg = env->NewStringUTF(e.what());
        env->CallVoidMethod(callback_global, onError_id, jmsg);
        env->DeleteLocalRef(jmsg);
    }

    LOGI("Inference thread finishing");
    if (smpl) llama_sampler_free(smpl);
    llama_batch_free(batch);
    env->DeleteGlobalRef(callback_global);
    g_is_generating = false;
    g_vm->DetachCurrentThread();
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_loadModel(JNIEnv* env, jobject thiz, jstring path, jint n_threads, jint n_ctx, jint n_batch) {
    std::lock_guard<std::mutex> lock(g_inference_mutex);
    const char* c_path = env->GetStringUTFChars(path, nullptr);
    std::string model_path(c_path);
    env->ReleaseStringUTFChars(path, c_path);

    LOGI("Loading model: %s", model_path.c_str());

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    g_state.model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!g_state.model) {
        LOGE("Failed to load model");
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    g_state.ctx = llama_init_from_model(g_state.model, ctx_params);
    if (!g_state.ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_state.model);
        g_state.model = nullptr;
        return 0;
    }

    g_state.previous_tokens.clear();
    return reinterpret_cast<jlong>(g_state.model);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_generate(JNIEnv* env, jobject thiz, jstring prompt, jobjectArray stop_strings_obj, jobject callback) {
    if (!g_state.ctx) return;
    g_state.stop_flag = true;

    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string s_prompt(c_prompt);
    env->ReleaseStringUTFChars(prompt, c_prompt);

    std::vector<std::string> stop_strings;
    if (stop_strings_obj != nullptr) {
        int count = env->GetArrayLength(stop_strings_obj);
        for (int i = 0; i < count; i++) {
            jstring stop_str_obj = (jstring)env->GetObjectArrayElement(stop_strings_obj, i);
            const char* stop_str_chars = env->GetStringUTFChars(stop_str_obj, nullptr);
            stop_strings.push_back(std::string(stop_str_chars));
            env->ReleaseStringUTFChars(stop_str_obj, stop_str_chars);
            env->DeleteLocalRef(stop_str_obj);
        }
    }

    jobject callback_global = env->NewGlobalRef(callback);

    LOGI("Starting inference thread");
    std::thread([s_prompt, stop_strings, callback_global]() {
        std::lock_guard<std::mutex> lock(g_inference_mutex);
        g_state.stop_flag = false;
        inference_thread(callback_global, s_prompt, stop_strings);
    }).detach();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_stop(JNIEnv* env, jobject thiz) {
    LOGI("Stop requested");
    g_state.stop_flag = true;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_unloadModel(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_inference_mutex);
    LOGI("Unloading model");
    if (g_state.ctx) {
        llama_free(g_state.ctx);
        g_state.ctx = nullptr;
    }
    if (g_state.model) {
        llama_model_free(g_state.model);
        g_state.model = nullptr;
    }
    g_state.previous_tokens.clear();
}
