package com.brahmadeo.tunedllm

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

interface LlmCallback {
    fun onToken(token: String)
    fun onStatus(status: String)
    fun onComplete()
    fun onError(message: String)
}

class LlmManager(private val context: Context) {
    private var nativePtr: Long = 0

    companion object {
        init {
            System.loadLibrary("llama_jni")
        }
    }

    external fun loadModel(path: String, nThreads: Int, nCtx: Int, nBatch: Int): Long
    external fun generate(
        prompt: String, 
        stopStrings: Array<String>, 
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        presencePenalty: Float,
        repetitionPenalty: Float,
        callback: LlmCallback
    )
    external fun stop()
    external fun clearContext()
    external fun unloadModel()

    suspend fun copyModelToInternalStorage(uri: Uri, onProgress: (Float) -> Unit): String? = withContext(Dispatchers.IO) {
        try {
            val destinationFile = File(context.filesDir, "active_model.gguf")
            
            val contentResolver = context.contentResolver
            val sourceSize = contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
            
            if (destinationFile.exists() && destinationFile.length() == sourceSize) {
                Log.i("LlmManager", "Model already exists in internal storage with correct size.")
                onProgress(1.0f)
                return@withContext destinationFile.absolutePath
            }

            contentResolver.openInputStream(uri)?.use { inputStream ->
                FileOutputStream(destinationFile).use { outputStream ->
                    val buffer = ByteArray(64 * 1024)
                    var bytesRead: Int
                    var totalBytesRead = 0L
                    while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                        totalBytesRead += bytesRead
                        if (sourceSize > 0) {
                            onProgress(totalBytesRead.toFloat() / sourceSize)
                        }
                    }
                }
            }
            return@withContext destinationFile.absolutePath
        } catch (e: Exception) {
            Log.e("LlmManager", "Failed to copy model: ${e.message}")
            null
        }
    }

    fun loadModelFromPath(path: String, nThreads: Int = 4, nCtx: Int = 8192, nBatch: Int = 512): Result<Long> {
        return try {
            nativePtr = loadModel(path, nThreads, nCtx, nBatch)
            if (nativePtr != 0L) {
                Result.success(nativePtr)
            } else {
                Result.failure(Exception("Native loadModel returned 0. Check logcat for LLM_JNI errors."))
            }
        } catch (e: Exception) {
            Log.e("LlmManager", "Error loading model: ${e.message}")
            Result.failure(e)
        }
    }
}
