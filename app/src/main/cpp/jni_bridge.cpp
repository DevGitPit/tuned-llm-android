#include <jni.h>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <chrono>
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
    openblas_set_num_threads(1);
#endif
    return JNI_VERSION_1_6;
}

static bool is_valid_utf8(const char* str) {
    if (!str) return false;
    const unsigned char* bytes = (const unsigned char*)str;
    while (*bytes) {
        if ((bytes[0] & 0x80) == 0x00) bytes++;
        else if ((bytes[0] & 0xE0) == 0xC0) { if ((bytes[1] & 0xC0) != 0x80) return false; bytes += 2; }
        else if ((bytes[0] & 0xF0) == 0xE0) { if ((bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80) return false; bytes += 3; }
        else if ((bytes[0] & 0xF8) == 0xF0) { if ((bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80) return false; bytes += 4; }
        else return false;
    }
    return true;
}

static jstring safe_new_string_utf(JNIEnv* env, const char* str) {
    if (!str || !*str) return env->NewStringUTF("");
    if (is_valid_utf8(str)) {
        jstring res = env->NewStringUTF(str);
        if (res) return res;
    }
    return env->NewStringUTF("");
}

static void batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < (size_t)seq_ids.size(); ++i) batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

static void inference_thread(jobject callback_global, std::string prompt_str, std::vector<std::string> stop_strings, 
                             float temp, float top_p, int top_k, float min_p, float presence_penalty, float repetition_penalty) {
    g_is_generating = true;
    JNIEnv* env;
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
        g_is_generating = false; return;
    }

    jclass callback_class = env->GetObjectClass(callback_global);
    jmethodID onToken_id = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
    jmethodID onStatus_id = env->GetMethodID(callback_class, "onStatus", "(Ljava/lang/String;)V");
    jmethodID onComplete_id = env->GetMethodID(callback_class, "onComplete", "()V");
    jmethodID onError_id = env->GetMethodID(callback_class, "onError", "(Ljava/lang/String;)V");

    const struct llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    llama_batch batch = llama_batch_init(llama_n_batch(g_state.ctx), 0, 1);
    llama_sampler* smpl = nullptr;

    try {
        if (!g_state.model || !g_state.ctx || !vocab) throw std::runtime_error("Engine not initialized");

        std::vector<llama_token> prompt_tokens(prompt_str.size() + 4);
        int n_prompt_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
        if (n_prompt_tokens < 0) throw std::runtime_error("Tokenization failed");
        prompt_tokens.resize(n_prompt_tokens);

        size_t n_past = 0;
        while (n_past < g_state.previous_tokens.size() && n_past < prompt_tokens.size() && g_state.previous_tokens[n_past] == prompt_tokens[n_past]) {
            n_past++;
        }

        llama_memory_t mem = llama_get_memory(g_state.ctx);
        if (n_past < g_state.previous_tokens.size()) {
            LOGI("Cache mismatch at %zu. Performing deep cache reset.", n_past);
            if (mem) {
                if (n_past == 0) llama_memory_clear(mem, true);
                else llama_memory_seq_rm(mem, -1, (llama_pos)n_past, -1);
            }
        }

        if (prompt_tokens.size() >= (size_t)llama_n_ctx(g_state.ctx)) throw std::runtime_error("Prompt too long for context window");

        // Initial status
        env->CallVoidMethod(callback_global, onStatus_id, env->NewStringUTF("Processing..."));

        int n_batch = llama_n_batch(g_state.ctx);
        std::vector<llama_token> session_tokens = prompt_tokens;

        auto pp_start = std::chrono::high_resolution_clock::now();
        int pp_tokens = 0;

        for (size_t i = n_past; i < prompt_tokens.size(); i += n_batch) {
            if (g_state.stop_flag.load()) break;
            int n_eval = (int)std::min((size_t)n_batch, prompt_tokens.size() - i);
            batch.n_tokens = 0;
            for (int j = 0; j < n_eval; ++j) {
                batch_add(batch, prompt_tokens[i + j], (llama_pos)(i + j), {0}, j == n_eval - 1);
            }
            if (llama_decode(g_state.ctx, batch) != 0) {
                LOGE("llama_decode error -1 at pos %zu. KV Cache might be corrupted.", i);
                throw std::runtime_error("Prompt decoding failed. Please try a new chat.");
            }
            
            pp_tokens += n_eval;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - pp_start).count();
            if (elapsed > 0.1) {
                char status_buf[64];
                snprintf(status_buf, sizeof(status_buf), "Processing... %.1f t/s", pp_tokens / elapsed);
                jstring jstatus = env->NewStringUTF(status_buf);
                env->CallVoidMethod(callback_global, onStatus_id, jstatus);
                env->DeleteLocalRef(jstatus);
            }
        }
        
        if (g_state.stop_flag.load()) {
            g_state.previous_tokens.clear();
            if (mem) llama_memory_clear(mem, true);
            throw std::runtime_error("Stopped");
        }

        n_past = prompt_tokens.size();
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(512, repetition_penalty, 0.0f, presence_penalty));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // Initial Generation status
        env->CallVoidMethod(callback_global, onStatus_id, env->NewStringUTF("Generating..."));

        int reasoning_tokens = 0;
        bool in_thinking = false;
        auto tg_start = std::chrono::high_resolution_clock::now();
        int tg_tokens = 0;

        while (n_past < (size_t)llama_n_ctx(g_state.ctx)) {
            if (g_state.stop_flag.load()) break;
            llama_token new_token_id = llama_sampler_sample(smpl, g_state.ctx, -1);
            session_tokens.push_back(new_token_id);
            if (llama_vocab_is_eog(vocab, new_token_id)) break;

            char buf[512];
            int n_piece = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            std::string piece(buf, n_piece > 0 ? n_piece : 0);

            if (piece.find("<think>") != std::string::npos || piece.find("<|channel>thought") != std::string::npos) in_thinking = true;
            if (in_thinking) reasoning_tokens++;
            if (piece.find("</think>") != std::string::npos || piece.find("<channel|>") != std::string::npos) in_thinking = false;
            
            if (reasoning_tokens > 4096) { 
                LOGW("Reasoning budget exceeded. Forcing answer.");
                break;
            }

            bool stop_found = false;
            for (const auto& s : stop_strings) {
                if (!s.empty() && piece.find(s) != std::string::npos) { stop_found = true; break; }
            }
            if (stop_found) break;

            if (!piece.empty()) {
                jstring jtoken = safe_new_string_utf(env, piece.c_str());
                env->CallVoidMethod(callback_global, onToken_id, jtoken);
                env->DeleteLocalRef(jtoken);
            }

            tg_tokens++;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - tg_start).count();
            if (elapsed > 0.1) {
                char status_buf[64];
                snprintf(status_buf, sizeof(status_buf), "Generating... %.1f t/s", tg_tokens / elapsed);
                jstring jstatus = env->NewStringUTF(status_buf);
                env->CallVoidMethod(callback_global, onStatus_id, jstatus);
                env->DeleteLocalRef(jstatus);
            }

            batch.n_tokens = 0;
            batch_add(batch, new_token_id, (llama_pos)n_past, {0}, true);
            if (llama_decode(g_state.ctx, batch) != 0) break;
            n_past++;
        }

        g_state.previous_tokens = session_tokens;
        if (!g_state.stop_flag.load()) env->CallVoidMethod(callback_global, onComplete_id);

    } catch (const std::exception& e) {
        g_state.previous_tokens.clear();
        jstring jmsg = env->NewStringUTF(e.what());
        env->CallVoidMethod(callback_global, onError_id, jmsg);
        env->DeleteLocalRef(jmsg);
    }

    if (smpl) llama_sampler_free(smpl);
    llama_batch_free(batch);
    env->DeleteGlobalRef(callback_global);
    g_is_generating = false;
    g_vm->DetachCurrentThread();
}

extern "C" JNIEXPORT jlong JNICALL Java_com_brahmadeo_tunedllm_LlmManager_loadModel(JNIEnv* env, jobject thiz, jstring path, jint n_threads, jint n_ctx, jint n_batch) {
    std::lock_guard<std::mutex> lock(g_inference_mutex);
    const char* c_path = env->GetStringUTFChars(path, nullptr);
    std::string model_path(c_path);
    env->ReleaseStringUTFChars(path, c_path);
    llama_model_params mp = llama_model_default_params();
    mp.use_mmap = true;
    g_state.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!g_state.model) return 0;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_batch; cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    g_state.ctx = llama_init_from_model(g_state.model, cp);
    g_state.previous_tokens.clear();
    return reinterpret_cast<jlong>(g_state.model);
}

extern "C" JNIEXPORT void JNICALL Java_com_brahmadeo_tunedllm_LlmManager_generate(
    JNIEnv* env, jobject thiz, jstring prompt, jobjectArray stop_strings_obj, 
    jfloat temp, jfloat top_p, jint top_k, jfloat min_p, jfloat presence_penalty, jfloat repetition_penalty, jobject callback) {
    if (!g_state.ctx) return;
    g_state.stop_flag = true;
    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string s_prompt(c_prompt);
    env->ReleaseStringUTFChars(prompt, c_prompt);
    std::vector<std::string> stop_strings;
    if (stop_strings_obj != nullptr) {
        int count = env->GetArrayLength(stop_strings_obj);
        for (int i = 0; i < count; i++) {
            jstring s = (jstring)env->GetObjectArrayElement(stop_strings_obj, i);
            const char* cs = env->GetStringUTFChars(s, nullptr);
            stop_strings.push_back(std::string(cs));
            env->ReleaseStringUTFChars(s, cs);
            env->DeleteLocalRef(s);
        }
    }
    jobject callback_global = env->NewGlobalRef(callback);
    std::thread([s_prompt, stop_strings, temp, top_p, top_k, min_p, presence_penalty, repetition_penalty, callback_global]() {
        std::lock_guard<std::mutex> lock(g_inference_mutex);
        g_state.stop_flag = false;
        inference_thread(callback_global, s_prompt, stop_strings, temp, top_p, top_k, min_p, presence_penalty, repetition_penalty);
    }).detach();
}

extern "C" JNIEXPORT void JNICALL Java_com_brahmadeo_tunedllm_LlmManager_stop(JNIEnv* env, jobject thiz) { g_state.stop_flag = true; }

extern "C" JNIEXPORT void JNICALL Java_com_brahmadeo_tunedllm_LlmManager_clearContext(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_inference_mutex);
    LOGI("Full hardware KV cache reset requested");
    g_state.previous_tokens.clear();
    if (g_state.ctx) {
        llama_memory_t mem = llama_get_memory(g_state.ctx);
        if (mem) llama_memory_clear(mem, true);
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_brahmadeo_tunedllm_LlmManager_unloadModel(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_inference_mutex);
    if (g_state.ctx) { llama_free(g_state.ctx); g_state.ctx = nullptr; }
    if (g_state.model) { llama_model_free(g_state.model); g_state.model = nullptr; }
    g_state.previous_tokens.clear();
}
