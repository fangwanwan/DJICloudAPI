package com.dji.sample.detection.service.impl;

import com.dji.sample.detection.service.LiveService;
import org.jetbrains.annotations.NotNull;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.OutputStream;

@Service
public class LiveServiceImpl implements LiveService {

    @Value("${ffmpeg.rtmp-url-start}")
    private String rtmpUrl;

    private VideoCapture video = new VideoCapture();

    @Override
    public void start() throws InterruptedException, IOException {
        Process ffmpeg = getProcess();
        OutputStream ffmpegInput = ffmpeg.getOutputStream();
        video.open(1);
        Mat image = new Mat();
        while(video.read(image)){
            // 编码为JPEG字节流
            MatOfByte mob = new MatOfByte();
            Imgcodecs.imencode(".jpg", image, mob);
            byte[] imageBytes = mob.toArray();

            // 写入FFmpeg stdin
            ffmpegInput.write(imageBytes);
            ffmpegInput.flush();
            // 控制帧率（可选）
            Thread.sleep(40); // 约15fps
        }
    }

    @Override
    public void stop() {
        video.release();
    }

    @NotNull
    private Process getProcess() throws IOException {
        ProcessBuilder pb = new ProcessBuilder(
                "ffmpeg",
                "-f", "image2pipe",         // 从管道读取图片流
                "-r", "15",                 // 帧率
                "-i", "-",                  // 输入为 stdin
                "-c:v", "libx264",          // 视频编码器
                "-preset", "ultrafast",     // 编码速度（低延迟）
                "-tune", "zerolatency",     // 零延迟优化
                "-c:a", "aac",              // 音频编码器（如果无音频可删除此行）
                "-f", "flv",                // 输出格式改为 FLV
                this.rtmpUrl  // 使用配置的RTMP地址
        );
        pb.redirectErrorStream(true);   // 合并错误输出
        Process ffmpeg = pb.start();
        return ffmpeg;
    }
}
