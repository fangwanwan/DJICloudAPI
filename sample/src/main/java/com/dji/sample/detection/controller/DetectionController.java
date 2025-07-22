package com.dji.sample.detection.controller;

import ai.onnxruntime.OrtException;
import com.dji.sample.detection.service.DetectionService;
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

    @GetMapping("people")
    public String DetectionPeople() throws IOException, OrtException {
        detectionService.detectPeople();
        return "开始识别行人";
    }
}
