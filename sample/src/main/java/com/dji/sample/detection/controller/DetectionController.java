package com.dji.sample.detection.controller;

import ai.onnxruntime.OrtException;
import com.dji.sample.detection.service.DetectionService;
import com.dji.sample.detection.service.LiveService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
@Slf4j
@RequestMapping("detection")
public class DetectionController {

    @Autowired
    private DetectionService detectionService;

    @Autowired
    private LiveService liveService;

    @GetMapping("people")
    public String DetectionPeople() throws IOException, OrtException, InterruptedException {
        detectionService.detectPeople();
        return "开始识别行人";
    }

    @GetMapping("stopDetect")
    public String StopDetect() {
        detectionService.stopDetect();
        return "停止识别";
    }

    @GetMapping("start")
    public String Start() throws IOException, InterruptedException {
        liveService.start();
        return "开始传输";
    }

    @GetMapping("stop")
    public String Stop(){
        liveService.stop();
        return "关闭传输";
    }
}
