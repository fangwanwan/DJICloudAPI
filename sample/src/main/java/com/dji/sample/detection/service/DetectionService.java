package com.dji.sample.detection.service;

import ai.onnxruntime.OrtException;

import java.io.IOException;

public interface DetectionService {
    void detectPeople() throws IOException, OrtException;
}
