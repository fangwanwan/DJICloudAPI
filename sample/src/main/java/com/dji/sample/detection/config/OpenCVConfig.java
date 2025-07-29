package com.dji.sample.detection.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;

@Slf4j
@Component
public class OpenCVConfig {

    @PostConstruct
    public void initOpenCV() {
        try {
            // 加载OpenCV本地库
            nu.pattern.OpenCV.loadLocally();
            log.info("OpenCV loaded successfully");
            // 处理操作系统特定的库加载
            loadOSSpecificLibraries();
        } catch (Exception e) {
            System.err.println("Failed to initialize OpenCV: " + e.getMessage());
            throw new RuntimeException("OpenCV initialization failed", e);
        }
    }

    private void loadOSSpecificLibraries() {
        String OS = System.getProperty("os.name").toLowerCase();
        try {
            if (OS.contains("win")) {
                System.load(ClassLoader.getSystemResource("lib/opencv_videoio_ffmpeg470_64.dll").getPath());
            }
        } catch (Exception e) {
            System.err.println("Failed to load OS specific library: " + e.getMessage());
        }
    }

    private void loadLibraryFromResource(String resourcePath) throws Exception {
        // 使用反射绕过Java的库加载限制
        Field field = ClassLoader.class.getDeclaredField("usr_paths");
        field.setAccessible(true);
        String[] paths = (String[]) field.get(null);

        // 检查库是否已加载
        for (String path : paths) {
            if (path.contains(resourcePath)) {
                return;
            }
        }

        // 加载库
        InputStream inputStream = getClass().getClassLoader().getResourceAsStream(resourcePath);
        if (inputStream == null) {
            throw new IOException("Library resource not found: " + resourcePath);
        }

        // 这里可以添加将资源复制到临时文件并加载的逻辑
        System.out.println("Loaded OS specific library: " + resourcePath);
    }
}
