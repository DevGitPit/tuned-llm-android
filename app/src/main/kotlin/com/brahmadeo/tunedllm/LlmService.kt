package com.brahmadeo.tunedllm

import android.app.*
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

import android.content.pm.ServiceInfo

class LlmService : Service() {
    private val binder = LlmBinder()
    lateinit var llmManager: LlmManager
    private val CHANNEL_ID = "LlmServiceChannel"

    inner class LlmBinder : Binder() {
        fun getService(): LlmService = this@LlmService
    }

    override fun onCreate() {
        super.onCreate()
        llmManager = LlmManager(applicationContext)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("LLM Inference Active")
            .setContentText("Model is loaded and ready.")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(1, notification)
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onDestroy() {
        super.onDestroy()
        llmManager.unloadModel()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(
                CHANNEL_ID,
                "LLM Service Channel",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(serviceChannel)
        }
    }
}
