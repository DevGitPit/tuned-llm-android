#include <jni.h>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <android/log.h>
#include "llama.h"

#define TAG "LLM_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::atomic<bool> stop_flag{false};
} g_state;

static JavaVM* g_vm = nullptr;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_vm = vm;
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

    return reinterpret_cast<jlong>(g_state.model);
}

static void inference_thread(jobject callback_global, std::string prompt_str) {
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

        // Reset memory for a new session
        llama_memory_t mem = llama_get_memory(g_state.ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }

        // Tokenize prompt
        std::vector<llama_token> prompt_tokens(prompt_str.size() + 2);
        int n_prompt_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
        if (n_prompt_tokens < 0) {
            throw std::runtime_error("Tokenization failed");
        }
        prompt_tokens.resize(n_prompt_tokens);
        LOGI("Prompt tokenized into %zu tokens", prompt_tokens.size());

        int n_past = 0;
        int n_batch = llama_n_batch(g_state.ctx);

        // Prompt Processing (KV Cache Fill)
        for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
            if (g_state.stop_flag.load()) break;

            int n_eval = (int)std::min((size_t)n_batch, prompt_tokens.size() - i);
            batch.n_tokens = 0;

            for (int j = 0; j < n_eval; ++j) {
                batch_add(batch, prompt_tokens[i + j], n_past + j, {0}, j == n_eval - 1);
            }

            int decode_res = llama_decode(g_state.ctx, batch);
            if (decode_res != 0) {
                LOGE("llama_decode failed during prompt processing with code: %d at n_past: %d", decode_res, n_past);
                throw std::runtime_error("llama_decode failed (" + std::to_string(decode_res) + ")");
            }
            n_past += n_eval;
        }

        // Generation Loop
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        LOGI("Starting generation loop at n_past: %d", n_past);

        while (n_past < llama_n_ctx(g_state.ctx)) {
            if (g_state.stop_flag.load()) break;
            llama_token new_token_id = llama_sampler_sample(smpl, g_state.ctx, -1);

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

            // Fallback string-based stop check for Gemma special tokens
            // Many models use variations of these with different IDs
            if (piece.find("<end_of_turn>") != std::string::npos || 
                piece.find("<eos>") != std::string::npos || 
                piece.find("</s>") != std::string::npos ||
                piece.find("<start_of_turn>") != std::string::npos ||
                piece.find("</start_of_turn>") != std::string::npos) {
                LOGI("String-based EOG detected: %s", piece.c_str());
                break;
            }

            // Filter out any turn tags if they somehow leaked past the stop check
            auto filter_tags = [&](std::string& s, const std::string& tag) {
                size_t pos = std::string::npos;
                while ((pos = s.find(tag)) != std::string::npos) {
                    s.erase(pos, tag.length());
                }
            };
            
            filter_tags(piece, "<start_of_turn>");
            filter_tags(piece, "</start_of_turn>");
            filter_tags(piece, "<end_of_turn>");

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
                LOGE("llama_decode failed during generation with code: %d at n_past: %d", decode_res, n_past);
                throw std::runtime_error("llama_decode failed (" + std::to_string(decode_res) + ")");
            }
            n_past++;
        }

        if (!g_state.stop_flag.load()) {
            env->CallVoidMethod(callback_global, onComplete_id);
        }

    } catch (const std::exception& e) {
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
Java_com_brahmadeo_tunedllm_LlmManager_generate(JNIEnv* env, jobject thiz, jstring prompt, jobject callback) {
    if (!g_state.ctx) return;

    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string s_prompt(c_prompt);
    env->ReleaseStringUTFChars(prompt, c_prompt);

    jobject callback_global = env->NewGlobalRef(callback);
    g_state.stop_flag = false;

    std::thread(inference_thread, callback_global, s_prompt).detach();
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
}
