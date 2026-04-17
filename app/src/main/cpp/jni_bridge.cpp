#include <jni.h>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <android/log.h>
#include "llama.h"

#ifdef GGML_USE_OPENBLAS
extern "C" {
    void openblas_set_num_threads(int num_threads);
}
#endif

#define TAG "LLM_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::atomic<bool> stop_flag{false};
    std::vector<llama_token> previous_tokens;
} g_state;

static JavaVM* g_vm = nullptr;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_vm = vm;
    
#ifdef GGML_USE_OPENBLAS
    LOGI("Setting OpenBLAS threads to 1 for better stability");
    openblas_set_num_threads(1);
#endif

    return JNI_VERSION_1_6;
}

// Helper to add a token to a batch
static void batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_loadModel(JNIEnv* env, jobject thiz, jstring path, jint n_threads, jint n_ctx, jint n_batch) {
    const char* c_path = env->GetStringUTFChars(path, nullptr);
    std::string model_path(c_path);
    env->ReleaseStringUTFChars(path, c_path);

    LOGI("Loading model from absolute path: %s", model_path.c_str());

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true; // CRITICAL: Use mmap for memory efficiency on limited RAM
    model_params.use_mlock = false;

    g_state.model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!g_state.model) {
        LOGE("Failed to load model from %s", model_path.c_str());
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

static void inference_thread(jobject callback_global, std::string prompt_str, std::vector<std::string> stop_strings) {
    JNIEnv* env;
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
        LOGE("Failed to attach current thread");
        return;
    }

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
        std::vector<llama_token> prompt_tokens(prompt_str.size() + 2);
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
            LOGI("Partial cache hit: n_past = %zu. Clearing stale tokens.", n_past);
            if (n_past == 0) {
                if (mem) llama_memory_clear(mem, true);
            } else {
                if (mem) llama_memory_seq_rm(mem, -1, (llama_pos)n_past, -1);
            }
        } else {
            LOGI("Cache hit: n_past = %zu. No clearing needed.", n_past);
        }

        int n_batch = llama_n_batch(g_state.ctx);
        std::vector<llama_token> session_tokens = prompt_tokens;

        // Prompt Processing (KV Cache Fill)
        bool prompt_interrupted = false;
        for (size_t i = n_past; i < prompt_tokens.size(); i += n_batch) {
            if (g_state.stop_flag.load()) {
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
                throw std::runtime_error("llama_decode failed (" + std::to_string(decode_res) + ")");
            }
        }
        
        if (prompt_interrupted) {
            // If we stopped mid-prompt, the cache is in an inconsistent state for the next run
            g_state.previous_tokens.clear();
            if (mem) llama_memory_clear(mem, true);
            throw std::runtime_error("Generation stopped");
        }

        n_past = prompt_tokens.size();

        // Generation Loop
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        LOGI("Starting generation loop at n_past: %zu", n_past);

        while (n_past < llama_n_ctx(g_state.ctx)) {
            if (g_state.stop_flag.load()) break;
            llama_token new_token_id = llama_sampler_sample(smpl, g_state.ctx, -1);
            
            session_tokens.push_back(new_token_id);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                LOGI("EOG token detected: %d", new_token_id);
                break;
            }

            // Convert token to piece
            char buf[256];
            int n_piece = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            std::string piece;
            if (n_piece < 0) {
                std::vector<char> dynamic_buf(-n_piece);
                int actual_piece = llama_token_to_piece(vocab, new_token_id, dynamic_buf.data(), dynamic_buf.size(), 0, true);
                piece = std::string(dynamic_buf.data(), actual_piece);
            } else {
                piece = std::string(buf, n_piece);
            }

            // Stop string check
            bool stop_found = false;
            for (const auto& stop_str : stop_strings) {
                if (piece.find(stop_str) != std::string::npos) {
                    LOGI("Stop string detected: %s", piece.c_str());
                    stop_found = true;
                    break;
                }
            }
            if (stop_found) break;

            if (!piece.empty()) {
                // JNI Callback
                jstring jtoken = env->NewStringUTF(piece.c_str());
                env->CallVoidMethod(callback_global, onToken_id, jtoken);
                env->DeleteLocalRef(jtoken);
            }

            // Feed Forward
            batch.n_tokens = 0;
            batch_add(batch, new_token_id, n_past, {0}, true);
            int decode_res = llama_decode(g_state.ctx, batch);
            if (decode_res != 0) {
                LOGE("llama_decode failed during generation with code: %d at n_past: %zu", decode_res, n_past);
                throw std::runtime_error("llama_decode failed (" + std::to_string(decode_res) + ")");
            }
            n_past++;
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

    // Cleanup
    if (smpl) llama_sampler_free(smpl);
    llama_batch_free(batch);
    env->DeleteGlobalRef(callback_global);
    g_vm->DetachCurrentThread();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_generate(JNIEnv* env, jobject thiz, jstring prompt, jobjectArray stop_strings_obj, jobject callback) {
    if (!g_state.ctx) return;

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
    g_state.stop_flag = false;

    std::thread(inference_thread, callback_global, s_prompt, stop_strings).detach();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_stop(JNIEnv* env, jobject thiz) {
    g_state.stop_flag = true;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_brahmadeo_tunedllm_LlmManager_unloadModel(JNIEnv* env, jobject thiz) {
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
