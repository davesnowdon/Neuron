package alpha2.uk.neuron;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.os.StrictMode;
import android.hardware.SensorEventListener;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.CameraBridgeViewBase;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import com.ubtechinc.alpha2ctrlapp.network.action.ClientAuthorizeListener;
import com.ubtechinc.alpha2robot.Alpha2RobotApi;
import com.ubtechinc.alpha2robot.constant.AlphaContant;
import com.ubtechinc.alpha2serverlib.interfaces.AlphaActionClientListener;
import com.ubtechinc.alpha2serverlib.interfaces.IAlpha2ActionListListener;
import com.ubtechinc.alpha2serverlib.interfaces.IAlpha2CustomMessageListener;
import com.ubtechinc.alpha2serverlib.interfaces.IAlpha2RobotClientListener;
import com.ubtechinc.alpha2serverlib.interfaces.IAlpha2RobotTextUnderstandListener;
import com.ubtechinc.alpha2serverlib.interfaces.IAlpha2SpeechGrammarInitListener;
import com.ubtechinc.alpha2serverlib.interfaces.IAlpha2SpeechGrammarListener;
import com.ubtechinc.alpha2serverlib.util.Alpha2SpeechMainServiceUtil;
import com.ubtechinc.contant.CustomLanguage;
import com.ubtechinc.contant.LauguageType;
import com.ubtechinc.contant.StaticValue;
import com.ubtechinc.developer.DeveloperAppStaticValue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Random;
import java.net.HttpURLConnection;
import java.net.URL;

public class MainActivity extends Activity implements
        CvCameraViewListener2, SensorEventListener, IAlpha2RobotClientListener, Alpha2SpeechMainServiceUtil.ISpeechInitInterface,
        IAlpha2RobotTextUnderstandListener, IAlpha2SpeechGrammarInitListener , IAlpha2ActionListListener , AlphaActionClientListener  {

    private Alpha2RobotApi mRobot;
    private ExitBroadcast mExitBroadcast;
    private SensorManager senSensorManager;
    private Sensor senAccelerometer;
    private long lastUpdate = 0;
    private float last_x, last_y, last_z;
    private static final int SHAKE_THRESHOLD = 1700;
    public String alphaposition = "stand";
    public String lastalphaposition = "stand";
    public long alphapositioncount = 0;
    public Boolean alphapositionsay = false;
    public boolean speaking = false;
    private boolean isOneAngle=true;
    URL url;
    HttpURLConnection conn;
    private String mPackageName;

    private static final String TAG                     = "Main::Activity";
    private static final Scalar    FACE_RECT_COLOR         = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR           = 0;
    public static final int        NATIVE_DETECTOR         = 1;

    private MenuItem mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType           = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize       = 0.2f;
    private int                    mAbsoluteFaceSize       = 0;

    public double                  headx                    =115;
    public double                  heady                    =120;
    public double                  headxlast                =115;
    public double                  headylast                =120;

    private CameraBridgeViewBase   mOpenCvCameraView;

    //Some UI items
    public TextView textView                = null;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status){
            switch(status){
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV Loaded Successfully");
                    System.loadLibrary("detection_based_tracker");

                    try{
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();
                    }catch(IOException e){
                        e.printStackTrace();
                        Log.i(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    //public static final int CAMERA_ID_BACK  = 99;
                    //public static final int CAMERA_ID_FRONT = 98;
                    //mOpenCvCameraView.setCameraIndex(98);
                    //mOpenCvCameraView.setCameraIndex(1);
                    //mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.enableView();

                } break;
                default:
                {
                    super.onManagerConnected(status);
                }break;
            }//switch
        }//onManagerConnected
    };//BaseLoaderCallback

    public MainActivity(){
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "JAVA";
        mDetectorName[NATIVE_DETECTOR] = "NATIVE (tracking)";
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
        StrictMode.setThreadPolicy(policy);
        senSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        senAccelerometer = senSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        senSensorManager.registerListener(this, senAccelerometer , SensorManager.SENSOR_DELAY_NORMAL);
        Context context = getApplicationContext();


        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        init();
    }

    public void init() {

        Bundle bundle = getIntent().getExtras();
        Log.i("zdy", "bundle " + bundle);

        mPackageName = this.getPackageName();
        IntentFilter filter = new IntentFilter();
        filter.addAction(DeveloperAppStaticValue.APP_EXIT);
        filter.addAction(mPackageName);
        filter.addAction(StaticValue.ALPHA_SPEECH_DIRECTION);
        filter.addAction(StaticValue.ALPHA_TTS_HINT);
        filter.addAction(DeveloperAppStaticValue.APP_ROBOT_UUID_INFO);
        filter.addAction(mPackageName + DeveloperAppStaticValue.APP_CONFIG);
        filter.addAction(mPackageName + DeveloperAppStaticValue.APP_CONFIG_SAVE);
        filter.addAction(mPackageName + DeveloperAppStaticValue.APP_BUTTON_EVENT);
        filter.addAction(mPackageName + DeveloperAppStaticValue.APP_BUTOON_EVENT_CLICK);
        mExitBroadcast = new ExitBroadcast();
        MainActivity.this.registerReceiver(mExitBroadcast, filter);
        String appkey = "222B998EDFA5FAD7FCE78678FB9F2521";
        mRobot = new Alpha2RobotApi(this, appkey,
                new ClientAuthorizeListener() {

                    @Override
                    public void onResult(int code, String info) {
                        // TODO Auto-generated method stub
                        Log.i("zdy", "code = " + code + " info= " + info);

                        mRobot.initSpeechApi(MainActivity.this,MainActivity.this);
                        mRobot.initActionApi(MainActivity.this);
                        mRobot.initChestSeiralApi();

                    }
                });

    }

    public void initOver() {
        // TODO Auto-generated method stub
        mRobot.speech_setVoiceName("xiaoyan");
        mRobot.speech_setRecognizedLanguage(LauguageType.LAU_ENGLISH);
        mRobot.speech_startRecognized("");
        mRobot.requestRobotUUID();
        Log.i("zdy", "Recognized initover");
        Calendar c = Calendar.getInstance();
        int currenthour = c.get(Calendar.HOUR_OF_DAY);
        if (currenthour < 12){
            say("good morning, powering up, please wait",false);
        } else if (currenthour < 17){
           say("good afternoon, powering up, please wait",false);
        } else {
            say("good evening, powering up, please wait",false);
        }
    }


    private Handler mHandler = new Handler() {

        @Override
        public void handleMessage(Message msg) {
            // TODO Auto-generated method stub
            super.handleMessage(msg);
            String text = (String) msg.obj;
            text = text.toLowerCase();
            if (text.contains("nlu_result:")){

            } else if (text.contains("Action_Performance")){
                mRobot.action_PlayActionName("Happy");
                int number = new Random().nextInt(10);
                String actionName = String.format("ACT%d", number);
                // mRobot.action_PlayActionName(actionName);
            } else {
                // Speech part

                if (text.contains(("tracking"))){
                    if (text.contains("off")){
                        say("Turning head tracking off",true);

                        if(mOpenCvCameraView != null)
                            mOpenCvCameraView.disableView();

                    }
                    if (text.contains("on")){
                        say("Turning head tracking on",true);
                        if(!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_11, MainActivity.this, mLoaderCallback)){

                        }
                    }
                }

                if (text.contains("light") || text.contains("lights")){
                    if (text.contains("off")){
                        mRobot.action_PlayActionName("Raise head");
                        try {
                            url = new URL("http://192.168.1.239:1001/light1off");

                            conn = (HttpURLConnection) url.openConnection();
                            conn.setInstanceFollowRedirects(true);
                            conn.connect();
                            conn.getResponseCode();
                            say("okay, turning the lights off",false);
                        } catch (IOException e){
                            say("sorry, I can not turn the lights off at the moment. There is a communication error it seems.",false);
                        }
//set the output to true, indicating you are outputting(uploading) POST data
                        conn.disconnect();
                    }
                    if (text.contains("on")){
                        mRobot.action_PlayActionName("Raise head");
                        try {
                            url = new URL("http://192.168.1.239:1001/light1on");
                            conn = (HttpURLConnection) url.openConnection();
                            conn.setInstanceFollowRedirects(true);
                            conn.connect();
                            conn.getResponseCode();
                            say("okay, turning the light on",false);
                        } catch (IOException e){
                            say("sorry, I can not turn the light on at the moment. There is a communication error it seems.",false);
                        }
//set the output to true, indicating you are outputting(uploading) POST data

                        conn.disconnect();
                    }
                }


                switch (text) {
                    case "can you see me":
                        say("Turning head tracking on",true);
                        if(!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_11, MainActivity.this, mLoaderCallback)){}
                        break;
                    case "close your eyes":
                        say("Turning head tracking off",true);
                        if(mOpenCvCameraView != null)mOpenCvCameraView.disableView();
                        break;
                    case "how are you doing":
                        say("I'm very well thank you for asking Steve.",false);
                        break;
                    case "play the radio":
                        mRobot.action_PlayActionName("Turn head leftward");
                        say("sorry I can't do that,,, but I know someone who can.",false);
                        mRobot.action_PlayActionName("Turn left");
                        say("Alexa,,, play radio one",false);
                        break;

                    case "thank you":
                        int number = new Random().nextInt(4);
                        if (number == 0){ say("You are welcome",false);}
                        if (number == 1){ say("I should think so too",false);}
                        if (number == 2){ say("no,, thank you",false);}
                        if (number == 3){ say("super duper, no problem",false);}
                        break;
                    case "okay thank you":
                        int numberr = new Random().nextInt(4);
                        if (numberr == 0){ say("You are welcome",false);}
                        if (numberr == 1){ say("I should think so too",false);}
                        if (numberr == 2){ say("no,, thank you",false);}
                        if (numberr == 3) {say("super duper, no problem",false);}
                        break;
                    case "can you walk forward for me":
                        mRobot.action_PlayActionName("fastwalk");
                        break;

                    default:
                      //  say("I'm sorry, I didn't quite catch that?",false);
                        break;
                }

            }

        }

    };

    @Override
    public void onServerPlayEnd(boolean isEnd) {
        Log.d("zdy", "onServerPlayEnd");
        speaking = false;
    }

    public void say(String whattosay, Boolean waitforend){
        speaking = true;
        mRobot.speech_StartTTS(whattosay);
       if (waitforend){
          while (speaking){}
        }
    }

    @Override
    public void onAlpha2UnderStandError(int arg0) {
        // TODO Auto-generated method stub
    }
    @Override
    public void speechGrammarInitCallback(String arg0, int nErrorCode) {
        // TODO Auto-generated method stub

        Log.i("zdy", "speeh_startGrammar init over");


        mHandler.obtainMessage(0).sendToTarget();

        mRobot.speeh_startGrammar(new IAlpha2SpeechGrammarListener() {

            @Override
            public void onSpeechGrammarResult(int SpeechResultType,
                                              String strResult) {
                Log.i("zdy", "SpeechResultType =" + SpeechResultType);
                Log.i("zdy", "strResult =" + strResult);
                mHandler.obtainMessage(1, strResult)
                        .sendToTarget();

            }

            @Override
            public void onSpeechGrammarError(int nErrorCode) {
                // TODO Auto-generated method stub

            }

        });

    }
    @Override
    public void onAlpha2UnderStandTextResult(String arg0) {
        // TODO Auto-generated method stub
        Log.i("zdy", "nlp result" + arg0);
        if (arg0 != null && !arg0.equals("")) {
            int number = new Random().nextInt(10);
            String actionName = String.format("ACT%d", number);
            mRobot.action_PlayActionName(actionName);
            String newText = new String(arg0);

        }

    }
    @Override  //This receives the text from speech
    public void onServerCallBack(String text) {
        // TODO Auto-generated method stub
        Log.i("zdy", "result" + text);

        // TODO Auto-generated method stub
        Log.i("zdy", "ʶ����" + text);
        if (text != null && !text.equals("")) {//

            String newText = new String(text);
            mRobot.speech_understandText(newText, this);
            mHandler.obtainMessage(2, text)
                    .sendToTarget();

        }

    }

    public class ExitBroadcast extends BroadcastReceiver {
        @Override
        public void onReceive(Context arg0, Intent intent) {
            // TODO Auto-generated method stub
            if (intent.getAction().equals(DeveloperAppStaticValue.APP_EXIT)) {
                Log.i("zdy", "speech_stopRecognized ");
                mRobot.releaseApi();
                mRobot = null;
                System.exit(0);
            } else if (intent.getAction().equals(mPackageName)) {

            }
        }
    }

    @Override
    public void onGetActionList(ArrayList<ArrayList<String>> list) {
        Message msg = new Message();
        msg.what = 5;
        msg.obj = list;
        mHandler.sendMessage(msg);

    }

    @Override
    public void onActionStop(String strActionFileName) {
        // TODO Auto-generated method stub

    }

    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor mySensor = sensorEvent.sensor;

        if (mySensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            float x = sensorEvent.values[0];
            float y = sensorEvent.values[1];
            float z = sensorEvent.values[2];

            long curTime = System.currentTimeMillis();


            if ((curTime - lastUpdate) > 100) {
                long diffTime = (curTime - lastUpdate);
                lastUpdate = curTime;

                float speed = Math.abs(x + y + z - last_x - last_y - last_z)/ diffTime * 10000;

                if (speed > SHAKE_THRESHOLD) {
                    if (!alphaposition.equals("shake")){

                        mRobot.action_PlayActionName("Blink");
                        say("oww that really hurts. I'm going all dizzy.",false);
                        alphaposition = "shake";
                        lastUpdate +=6000;
                    }

                } else {
                    //Alpha position
                    if (y < -9.5 & (z < 2)){
                        alphaposition = "front";
                    }

                    if (y > 9.5 & (z < 2)){
                        alphaposition = "back";
                    }
                    if (x > 9.5 & (z < 1.5)){
                        alphaposition = "left";
                    }
                    if (x < -9.5 & (z < 1.5)){
                        alphaposition = "right";
                    }
                    if (z < -9.5) { //reset alpha position
                        alphaposition = "upsidedown";
                    }
                }

                if (lastalphaposition.equals(alphaposition)){
                    alphapositioncount +=1;
                } else {
                    alphapositioncount = 0;
                    lastalphaposition = alphaposition;
                    alphapositionsay = false;
                }

                if (alphapositioncount > 17){

                    if (!alphapositionsay){
                        switch(alphaposition){
                            case "back":
                                say("Why am I lying down?",true);
                                say("give me a minute,,,, I'll get up",true);
                                mRobot.action_PlayActionName("BackStand");
                                break;
                            case "front":
                                say("Hey, I can not see,, what's going on",true);
                                say("wait a minute,,,, I'll try and get up",true);
                                mRobot.action_PlayActionName("FrontStand");
                                break;
                            case "upsidedown":
                                say("please put me down. are you stupid? this is not how you hold me.",false);
                            case "":
                                break;
                            case "shake":

                                break;

                            default:
                                say("I'm on my " + alphaposition,false);
                                break;
                        }

                        alphapositionsay = true;
                    }
                    alphapositionsay = true;
                } else {
                    alphapositionsay = false;
                }

                if (x > -2 & x < 2 & y > -2 & y < 2 ){ //reset alpha position
                    if (z > 8){
                        alphaposition = "";
                        alphapositioncount= 0;
                        alphapositionsay = false;
                    }

                }

                last_x = x;
                last_y = y;
                last_z = z;
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    protected void onPause() {
        super.onPause();

        senSensorManager.unregisterListener(this);

        if(mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

    }
    protected void onResume() {
        super.onResume();
        senSensorManager.registerListener(this, senAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);

        if(!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_11, this, mLoaderCallback)){
            Log.i(TAG, "OpenCVLoader Failed on Resume");
        }

    }

    @Override
    protected void onDestroy() {
        // TODO Auto-generated method stub
        super.onDestroy();
        if (mExitBroadcast != null) {
            this.unregisterReceiver(mExitBroadcast);
            mExitBroadcast = null;
        }
        /**
         * Before destroy, stop TTS and action.
         */
        if (mRobot != null) {
            mRobot.speech_StopTTS();
            mRobot.action_StopAction();
        }

        if (mRobot != null) {
            mRobot.releaseApi();
            mRobot = null;
        }
        Log.i("zdy", "onDestroy ");
        mOpenCvCameraView.disableView();

    }
    public void onCameraViewStarted(int width, int height){
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped(){
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();


        //Core.transpose(mRgba, mGray);
        //Camera Upside down Fix...
        //Core.flip(mRgba, mGray, -1);

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();


        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

        if (facesArray.length > 0) { // location of center of square
            double headxtemp = facesArray[0].x + (facesArray[0].width / 2);
            double headytemp = facesArray[0].y + (facesArray[0].height / 2);
            headx = (headxtemp / 1280) * 100;
            heady = (headytemp / 720) * 100;
            short time = 380;

            if (headx > 50){ // look left
                if (headx > 53){headxlast +=1;}
                if (headx > 60){headxlast +=2;}
                if (headx > 70){headxlast +=3;}
                if (headx > 80){headxlast +=4;}
            } else {
                if (headx < 47){headxlast -=1;}
                if (headx < 40){headxlast -=2;}
                if (headx < 30){headxlast -=3;}
                if (headx < 20){headxlast -=4;}
            }

            if (heady > 50){ // look left
                if (heady > 55){headylast +=1;}
                if (heady > 60){headylast +=1;}
                if (heady > 70){headylast +=2;}
                if (heady > 80){headylast +=2;}
            } else {
                if (heady < 45){headylast -=1;}
                if (heady < 40){headylast -=1;}
                if (heady < 30){headylast -=2;}
                if (heady < 20){headylast -=2;}
            }

            if (headxlast > 170){headxlast  = 170;}
            if (headxlast < 70){headxlast  = 70;}
            if (headylast > 155){headylast = 155;}
            if (headylast < 100){headylast = 100;}

            mRobot.chest_SendOneFreeAngle((byte) 19, (int) headxlast, time);
            mRobot.chest_SendOneFreeAngle((byte) 20, (int) headylast + 5, time); // The +5 makes alpha look down slightly, improved voice control
        }

        return mRgba;
    }

    private void setMinFaceSize(float faceSize){
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }

}
