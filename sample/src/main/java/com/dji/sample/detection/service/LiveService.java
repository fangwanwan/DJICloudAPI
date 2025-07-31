package com.dji.sample.detection.service;

import org.springframework.stereotype.Service;

import java.io.IOException;


public interface LiveService {
     void start() throws InterruptedException, IOException;

    void stop();
}
