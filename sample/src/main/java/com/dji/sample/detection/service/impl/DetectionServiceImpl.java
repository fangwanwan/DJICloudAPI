package com.dji.sample.detection.service.impl;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.dji.sample.detection.service.DetectionService;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.springframework.stereotype.Service;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.dji.sample.detection.config.ODConfig;
import com.dji.sample.detection.domain.ODResult;
import com.dji.sample.detection.utils.ImageUtil;
import com.dji.sample.detection.utils.Letterbox;
import org.jetbrains.annotations.NotNull;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.MatOfByte;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;

import org.springframework.beans.factory.annotation.Value;
import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

@Slf4j
@Service
public class DetectionServiceImpl implements DetectionService {
    @Value("${yolo.model-path}")
    private String modelPath;

    @Value("${ffmpeg.rtmp-url}")
    private String rtmpUrl;

    @Value("${video.source}")
    private int videoSource;

    @Value("${video.default-path}")
    private String defaultVideoPath;

    // 跳帧检测，一般设置为3，毫秒内视频画面变化是不大的，快了无意义，反而浪费性能
    @Value("${detection.skip-frames}")
    int detect_skip;

    private OrtSession session;
    private String[] labels;
    private Process ffmpeg;
    private OutputStream ffmpegInput;
    OrtEnvironment environment;
    ODConfig odConfig = new ODConfig();

    @PostConstruct
    public void init() throws IOException, OrtException {
        // 初始化标签
        labels = new String[]{
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        };

        // 加载ONNX模型
        loadONNXModel();

        // 初始化FFmpeg
        initFFmpeg();
    }

    private void loadONNXModel() throws OrtException {
        environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        // 实际项目中建议开启GPU加速
        // sessionOptions.addCUDA(0);
        session = environment.createSession(modelPath, sessionOptions);

        // 输出模型信息
        logModelInfo();
    }

    private void logModelInfo() throws OrtException {
        session.getInputInfo().keySet().forEach(x -> {
            try {
                log.info("input name ={}",x);
                log.info(session.getInputInfo().get(x).getInfo().toString());
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        });
    }

    private void initFFmpeg() throws IOException {
        log.info("FFmpeg 支持: " + Videoio.CAP_FFMPEG);
        ffmpeg = getProcess();
        ffmpegInput = ffmpeg.getOutputStream();
    }

    public void detectPeople() throws IOException, OrtException {
        // 加载标签及颜色
        odConfig = new ODConfig();
        VideoCapture video = new VideoCapture();

        // 打开视频源
        openVideoSource(video);
    }

    private void openVideoSource(VideoCapture video) throws IOException {
        try {
            // 尝试打开视频源
            video.open(videoSource);
        } catch (Exception e) {
            System.err.println("视频源初始化失败: " + e.getMessage());
            throw new RuntimeException("无法初始化视频源", e);
        }

        // 在这里先定义下框的粗细、字的大小、字的类型、字的颜色(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min((int)video.get(Videoio.CAP_PROP_FRAME_WIDTH), (int)video.get(Videoio.CAP_PROP_FRAME_HEIGHT));
        int thickness = minDwDh / ODConfig.lineThicknessRatio;
        double fontSize = minDwDh / ODConfig.fontSizeRatio;
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

        Mat img = new Mat();

        // 跳帧计数
        int detect_skip_index = 1;

        // 最新一帧也就是上一帧推理结果
        float[][] outputData   = null;

        //当前最新一帧。上一帧也可以暂存一下
        Mat image;

        Letterbox letterbox = new Letterbox();
        OnnxTensor tensor = null;
        OrtSession.Result output = null;
        // 使用多线程和GPU可以提升帧率，线上项目必须多线程！！！,一个线程拉流，将图像存到[定长]队列或数组或者集合，一个线程模型推理，中间通过变量或者队列交换数据,代码示例仅仅使用单线程
        while (video.read(img)) {
            if ((detect_skip_index % detect_skip == 0) || outputData == null){
                try{
                    image = img.clone();
                    image = letterbox.letterbox(image);
                    Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);

                    image.convertTo(image, CvType.CV_32FC1, 1. / 255);
                    float[] whc = new float[3 * 640 * 640];
                    image.get(0, 0, whc);
                    float[] chw = ImageUtil.whc2cwh(whc);

                    detect_skip_index = 1;

                    FloatBuffer inputBuffer = FloatBuffer.wrap(chw);
                    tensor = OnnxTensor.createTensor(environment, inputBuffer, new long[]{1, 3, 640, 640});

                    HashMap<String, OnnxTensor> stringOnnxTensorHashMap = new HashMap<>();
                    stringOnnxTensorHashMap.put(session.getInputInfo().keySet().iterator().next(), tensor);

                    // 运行推理
                    // 模型推理本质是多维矩阵运算，而GPU是专门用于矩阵运算，占用率低，如果使用cpu也可以运行，可能占用率100%属于正常现象，不必纠结。
                    output = session.run(stringOnnxTensorHashMap);

                    // 得到结果,缓存结果
                    outputData = (float[][]) output.get(0).getValue();
                }catch (OrtException e){
                    System.err.println("推理出错"+e.getMessage());
                }finally {
                    if (output != null) {
                        output.close();
                    }
                    if (tensor != null) {
                        tensor.close();
                    }
                }
            }else{
                detect_skip_index = detect_skip_index + 1;
            }

            for(float[] x : outputData){

                ODResult odResult = new ODResult(x);
                // 业务逻辑写在这里，注释下面代码，增加自己的代码，根据返回识别到的目标类型，编写告警逻辑。等等
                // 实际项目中建议不要在视频画面上画框和文字，只告警，或者在告警图片上画框。画框和文字对视频帧率影响非常大
                // 画框
                Point topLeft = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio());
                Point bottomRight = new Point((odResult.getX1() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY1() - letterbox.getDh()) / letterbox.getRatio());
                Scalar color = new Scalar(odConfig.getOtherColor(odResult.getClsId()));

                Imgproc.rectangle(img, topLeft, bottomRight, color, thickness);
                // 框上写文字
                String boxName = labels[odResult.getClsId()];
                Point boxNameLoc = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio() - 3);

                // 也可以二次往视频画面上叠加其他文字或者数据，比如物联网设备数据等等
                Imgproc.putText(img, boxName, boxNameLoc, fontFace, 0.7, color, thickness);
                // System.out.println(odResult+"   "+ boxName);

            }

            // 编码为JPEG字节流
            MatOfByte mob = new MatOfByte();
            Imgcodecs.imencode(".jpg", img, mob);
            byte[] imageBytes = mob.toArray();

            // 写入FFmpeg stdin
            ffmpegInput.write(imageBytes);
            ffmpegInput.flush();
            // 控制帧率（可选）
            //Thread.sleep(66); // 约15fps
        }

        HighGui.destroyAllWindows();
        video.release();
    }

    @PreDestroy
    public void destroy() {
        // 资源清理
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                System.err.println("OrtSession关闭失败: " + e.getMessage());
            }
        }
        if (ffmpeg != null) {
            ffmpeg.destroy();
        }
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
