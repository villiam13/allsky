#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "include/ASICamera2.h"
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <errno.h>
#include <string>
#include <iostream>
#include <cstdio>
#include <tr1/memory>
#include <ctime>
#include <stdlib.h>
#include <signal.h>
#include <fstream>
#include <syslog.h>
#include <stdarg.h>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

using namespace cv;

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

cv::Mat pRgb;
char nameCnt[128];
char const *fileName = "image.jpg";
std::vector<int> compression_parameters;
bool bMain = true, bDisplay = false;
std::string dayOrNight;

bool bSaveRun = false, bSavingImg = false;
pthread_mutex_t mtx_SaveImg;
pthread_cond_t cond_SatrtSave;
volatile long saveExpTime = 0;

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

void cvText(cv::Mat &img, const char *text, int x, int y, double fontsize, int linewidth, int linetype, int fontname,
            int fontcolor[], int imgtype, int outlinefont)
{
    if (imgtype == ASI_IMG_RAW16)
    {
        if (outlinefont)
            cv::putText(img, text, cvPoint(x, y), fontname, fontsize, cvScalar(0,0,0), linewidth+4, linetype);
        cv::putText(img, text, cvPoint(x, y), fontname, fontsize, cvScalar(fontcolor[0], fontcolor[1], fontcolor[2]),
                    linewidth, linetype);
    }
    else
    {
        if (outlinefont)
            cv::putText(img, text, cvPoint(x, y), fontname, fontsize, cvScalar(0,0,0, 255), linewidth+4, linetype);
        cv::putText(img, text, cvPoint(x, y), fontname, fontsize,
                    cvScalar(fontcolor[0], fontcolor[1], fontcolor[2], 255), linewidth, linetype);
    }
}

void a_logger(const char *format, ...)
{
    char c_buff[1000];

    va_list arg;
    va_start(arg, format);
    vsprintf (c_buff, format, arg);
    va_end (arg);

    printf("%s", c_buff);
    syslog(LOG_INFO, c_buff);
}

char *getTime()
{
    static int seconds_last = 99;
    static char TimeString[128];
    timeval curTime;
    gettimeofday(&curTime, NULL);
    if (seconds_last == curTime.tv_sec)
    {
        return 0;
    }

    seconds_last = curTime.tv_sec;
    strftime(TimeString, 80, "%Y%m%d %H:%M:%S", localtime(&curTime.tv_sec));
    return TimeString;
}

std::string exec(const char *cmd)
{
    std::tr1::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
        return "ERROR";
    char buffer[128];
    std::string result = "";
    while (!feof(pipe.get()))
    {
        if (fgets(buffer, 128, pipe.get()) != NULL)
        {
            result += buffer;
        }
    }
    return result;
}

void *Display(void *params)
{
    cv::Mat *pImg = (cv::Mat *)params;
    cvNamedWindow("video", 1);
    while (bDisplay)
    {
        cvShowImage("video", pImg);
        cvWaitKey(100);
    }
    cvDestroyWindow("video");
    a_logger("Display thread over\n");
    return (void *)0;
}

void *SaveImgThd(void *para)
{
    while (bSaveRun)
    {
        pthread_mutex_lock(&mtx_SaveImg);
        pthread_cond_wait(&cond_SatrtSave, &mtx_SaveImg);
        bSavingImg = true;
        if (pRgb.data)
        {
            imwrite(fileName, pRgb, compression_parameters);
            char buf[255];
            if (dayOrNight == "NIGHT")
            {
                sprintf(buf, "scripts/saveImageNight.sh %ld &", saveExpTime);
            }
            else
            {
                sprintf(buf, "scripts/saveImageDay.sh %ld &", saveExpTime);
            }
            system(buf);
        }
        bSavingImg = false;
        pthread_mutex_unlock(&mtx_SaveImg);
    }

    a_logger("save thread over\n");
    return (void *)0;
}

void IntHandle(int i)
{
    bMain = false;
}

void calculateDayOrNight(const char *latitude, const char *longitude, const char *angle)
{
    char sunwaitCommand[128];
    sprintf(sunwaitCommand, "sunwait poll exit set angle %s %s %s", angle, latitude, longitude);
    dayOrNight = exec(sunwaitCommand);
    dayOrNight.erase(std::remove(dayOrNight.begin(), dayOrNight.end(), '\n'), dayOrNight.end());
}

void writeToLog(int val)
{
    std::ofstream outfile;
    outfile.open("log.txt", std::ios_base::app);
    outfile << val;
    outfile << "\n";
}

/**
Apply a Sobel edge detection transform to help better spot the subtle lines
when the mosaic effect occurs

See https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
for the refrence code
**/
cv::Mat applySobel(cv::Mat &src)
{
    // Variables
    cv::Mat src_gray;
	cv::Mat sobel;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

    // Perform a Sobel transform
    cv::GaussianBlur( src, src, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    // Convert it to gray (using CV_BGR2GRAY as that seems to work)
	cv::cvtColor( src, src_gray, CV_BGR2GRAY );

    // Generate grad_x and grad_y
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
	cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
	cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	cv::convertScaleAbs( grad_y, abs_grad_y );

    // Total Gradient (approximate)
    // We set thr weight of the grad_x to 0 because in my camera's case those would
    // be vertical lines
	cv::addWeighted( abs_grad_x, 0.0, abs_grad_y, 0.8, 0, sobel );

    // Return the image
    return sobel;
}

/**
Some ASi120MC clone cameras (and perhaps others) will occasionaly produce an image
which is comprised of multiple rectangles of parts of the scene. This function attempts
to detect if there is an horizontal line cutting across the image and if there is it will
return true.

See bug_samples/mosaic-image.jpg for an example of the problem

See https://www.codepool.biz/opencv-line-detection.html and https://stackoverflow.com/a/7228823
for info about detecting horizontal lines
**/
bool mosaicImage(cv::Mat &src)
{
    a_logger("Checking if this is a \"mosaic\" image.\n");
    cv::Mat img_src = src.clone();

    cv::Mat src_sobel = applySobel(img_src);

    // Usefull for debugging, gives us a copy of the image
    //imwrite("mosaic.jpg", src_sobel);

    cv::Mat img_canny, img_gray;

    // Detect edges
    cv::Canny(src_sobel, img_canny, 50, 200, 3);

    std::vector<Vec4i> lines;
	cv::HoughLinesP(img_canny, lines, 1, CV_PI / 2, 50, 50, 10);
    //a_logger("Lines: %d\n", lines.size());

    if (lines.size() > 0) {
        a_logger("We found some horizontal lines.\n");

        // useful for debugging
        //img_copy = img_canny.clone();

        /*for (size_t i = 0; i < lines.size(); i++)
	    {
		    Vec4i l = lines[i];
		    line(img_copy, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, 2);
	    }

	    imwrite("horizontal.jpg", img_copy); */

        return true;
    }

    
    return false;
}

bool brokenDetector(cv::Mat &src)
{
    int i, j, c;

    int ddepth = -1;
    cv::Point anchor = cv::Point( -1, -1 );
    double delta = 0;
    float filter[3][3] = {{1,1,1}, {0,0,0}, {-1,-1,-1}};
    cv::Mat kernel = cv::Mat( 3, 3, CV_32F, filter );
    // Apply filter
    cv::Mat dst;
    cv::filter2D(src, dst, ddepth , kernel, anchor, delta, cv::BORDER_DEFAULT );

    cv::resize(dst, dst, cv::Size(16, pRgb.rows), 0, 0, cv::INTER_AREA);
    // imwrite("out.jpg", dst);
    for (i = 0; i < dst.rows; i++)
    {
        c = 0;
        for (j = 0; j < dst.cols; j++)
        {
            if (dst.at<uint8_t>(i,j) > 80)
            {
                // a_logger("%dx%d: %d\n", i, j, dst.at<uint8_t>(i,j));
                c ++;
            }
            else
            {
                c = 0;
            }
            if (c > dst.cols / 2)
            {
                a_logger("Broken Image detected! %d at %d line\n", c, i);
                return true;
            }
        }
    }
    return false;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    signal(SIGINT, IntHandle);
    pthread_mutex_init(&mtx_SaveImg, 0);
    pthread_cond_init(&cond_SatrtSave, 0);

    int fontname[] = { CV_FONT_HERSHEY_SIMPLEX,        CV_FONT_HERSHEY_PLAIN,         CV_FONT_HERSHEY_DUPLEX,
                       CV_FONT_HERSHEY_COMPLEX,        CV_FONT_HERSHEY_TRIPLEX,       CV_FONT_HERSHEY_COMPLEX_SMALL,
                       CV_FONT_HERSHEY_SCRIPT_SIMPLEX, CV_FONT_HERSHEY_SCRIPT_COMPLEX };
    int fontnumber = 0;
    int iStrLen, iTextX = 15, iTextY = 25;
    char const *ImgText   = "";
    double fontsize       = 0.6;
    int linewidth         = 1;
    int outlinefont       = 0;
    int fontcolor[3]      = { 255, 0, 0 };
    int smallFontcolor[3] = { 0, 0, 255 };
    int linetype[3]       = { CV_AA, 8, 4 };
    int linenumber        = 0;

    char buf[1024]    = { 0 };
    char bufTime[128] = { 0 };
    char bufTemp[128] = { 0 };

    int width             = 0;
    int height            = 0;
    int bin               = 1;
    int Image_type        = 1;
    int asiBandwidth      = 40;
    int asiExposure       = 5000000;
    int asiMaxExposure    = 10000;
    int asiAutoExposure   = 0;
    int asiGain           = 150;
    int asiMaxGain        = 200;
    int asiAutoGain       = 0;
    int delay             = 10;   // Delay in milliseconds. Default is 10ms
    int daytimeDelay      = 5000; // Delay in milliseconds. Default is 5000ms
    int asiWBR            = 65;
    int asiWBB            = 85;
    int asiGamma          = 50;
    int asiBrightness     = 50;
    int asiFlip           = 0;
    int asiCoolerEnabled  = 0;
    long asiTargetTemp    = 0;
    char const *latitude  = "60.7N"; //GPS Coordinates of Whitehorse, Yukon where the code was created
    char const *longitude = "135.05W";
    char const *angle  = "-6"; // angle of the sun with the horizon (0=sunset, -6=civil twilight, -12=nautical twilight, -18=astronomical twilight)
    int preview        = 0;
    int time           = 1;
    int darkframe      = 0;
    int showDetails    = 0;
    int daytimeCapture = 0;
    int help           = 0;
    int quality        = 200;

    char const *bayer[] = { "RG", "BG", "GR", "GB" };
    int CamNum          = 0;
    int i;
    void *retval;
    bool endOfNight    = false;
    pthread_t hthdSave = 0;

    //-------------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------------------
    a_logger("\n");
    a_logger("%s ******************************************\n", KGRN);    
    a_logger("%s *** Allsky Camera Software v0.6 | 2019 ***\n", KGRN);    
    a_logger("%s ******************************************\n\n", KGRN);    
    a_logger("\%sCapture images of the sky with a Raspberry Pi and an ASI Camera\n", KGRN);    
    a_logger("\n");
    a_logger("%sAdd -h or -help for available options \n", KYEL);    
    a_logger("\n");
    a_logger("\%sAuthor: ", KNRM);    
    a_logger("Thomas Jacquin - <jacquin.thomas@gmail.com>\n\n");
    a_logger("\%sContributors:\n", KNRM);    
    a_logger("-Knut Olav Klo\n");
    a_logger("-Daniel Johnsen\n");
    a_logger("-Yang and Sam from ZWO\n");
    a_logger("-Robert Wagner\n");
    a_logger("-Michael J. Kidd - <linuxkidd@gmail.com>\n\n");

    if (argc > 0)
    {
        for (i = 0; i < argc - 1; i++)
        {
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0)
            {
                help = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-width") == 0)
            {
                width = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-height") == 0)
            {
                height = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-type") == 0)
            {
                Image_type = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-quality") == 0)
            {
                quality = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-exposure") == 0)
            {
                asiExposure = atoi(argv[i + 1]) * 1000;
                i++;
            }
            else if (strcmp(argv[i], "-maxexposure") == 0)
            {
                asiMaxExposure = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-autoexposure") == 0)
            {
                asiAutoExposure = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-gain") == 0)
            {
                asiGain = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-maxgain") == 0)
            {
                asiMaxGain = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-autogain") == 0)
            {
                asiAutoGain = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-gamma") == 0)
            {
                asiGamma = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-brightness") == 0)
            {
                asiBrightness = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-bin") == 0)
            {
                bin = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-delay") == 0)
            {
                delay = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-daytimeDelay") == 0)
            {
                daytimeDelay = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-wbr") == 0)
            {
                asiWBR = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-wbb") == 0)
            {
                asiWBB = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-text") == 0)
            {
                ImgText = (argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-textx") == 0)
            {
                iTextX = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-texty") == 0)
            {
                iTextY = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-fontname") == 0)
            {
                fontnumber = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-fontcolor") == 0)
            {
                fontcolor[0] = atoi(argv[i + 1]);
                i++;
                fontcolor[1] = atoi(argv[i + 1]);
                i++;
                fontcolor[2] = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-smallfontcolor") == 0)
            {
                smallFontcolor[0] = atoi(argv[i + 1]);
                i++;
                smallFontcolor[1] = atoi(argv[i + 1]);
                i++;
                smallFontcolor[2] = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-fonttype") == 0)
            {
                linenumber = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-fontsize") == 0)
            {
                fontsize = atof(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-fontline") == 0)
            {
                linewidth = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-outlinefont") == 0)
            {
                outlinefont = atoi(argv[i + 1]);
				if (outlinefont != 0)
				    outlinefont = 1;
                i++;
            }
            else if (strcmp(argv[i], "-flip") == 0)
            {
                asiFlip = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-usb") == 0)
            {
                asiBandwidth = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-filename") == 0)
            {
                fileName = (argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-latitude") == 0)
            {
                latitude = argv[i + 1];
                i++;
            }
            else if (strcmp(argv[i], "-longitude") == 0)
            {
                longitude = argv[i + 1];
                i++;
            }
            else if (strcmp(argv[i], "-angle") == 0)
            {
                angle = argv[i + 1];
                i++;
            }
            else if (strcmp(argv[i], "-preview") == 0)
            {
                preview = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-time") == 0)
            {
                time = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-darkframe") == 0)
            {
                darkframe = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-showDetails") == 0)
            {
                showDetails = atoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "-daytime") == 0)
            {
                daytimeCapture = atoi(argv[i + 1]);
                i++;
            }
	    else if (strcmp(argv[i], "-coolerEnabled") == 0)
            {
                asiCoolerEnabled = atoi(argv[i + 1]);
                i++;
            }
	    else if (strcmp(argv[i], "-targetTemp") == 0)
            {
                asiTargetTemp = atol(argv[i + 1]);
                i++;
            }
        }
    }

    if (help == 1)
    {
        printf("%sAvailable Arguments: \n", KYEL);
        printf(" -width                             - Default = Camera Max Width \n");
        printf(" -height                            - Default = Camera Max Height \n");
        printf(" -exposure                          - Default = 5000000 - Time in µs (equals to 5 sec) \n");
        printf(" -maxexposure                       - Default = 10000 - Time in ms (equals to 10 sec) \n");
        printf(" -autoexposure                      - Default = 0 - Set to 1 to enable auto Exposure \n");
        printf(" -gain                              - Default = 50 \n");
        printf(" -maxgain                           - Default = 200 \n");
        printf(" -autogain                          - Default = 0 - Set to 1 to enable auto Gain \n");
        printf(" -coolerEnabled                     - Set to 1 to enable cooler (works on cooled cameras only) \n");
        printf(" -targetTemp                        - Target temperature in degrees C (works on cooled cameras only) \n");
	    printf(" -gamma                             - Default = 50 \n");
        printf(" -brightness                        - Default = 50 \n");
        printf(" -wbr                               - Default = 50   - White Balance Red \n");
        printf(" -wbb                               - Default = 50   - White Balance Blue \n");
        printf(" -bin                               - Default = 1    - 1 = binning OFF (1x1), 2 = 2x2 binning, 4 = 4x4 "
               "binning\n");
        printf(" -delay                             - Default = 10   - Delay between images in milliseconds - 1000 = 1 "
               "sec.\n");
        printf(" -daytimeDelay                      - Default = 5000   - Delay between images in milliseconds - 5000 = "
               "5 sec.\n");
        printf(" -type = Image Type                 - Default = 0    - 0 = RAW8,  1 = RGB24,  2 = RAW16 \n");
        printf(" -quality                           - Default PNG=3, JPG=95, Values: PNG=0-9, JPG=0-100\n");
        printf(" -usb = USB Speed                   - Default = 40   - Values between 40-100, This is "
               "BandwidthOverload \n");
        printf(" -filename                          - Default = IMAGE.PNG \n");
        printf(" -flip                              - Default = 0    - 0 = Orig, 1 = Horiz, 2 = Verti, 3 = Both\n");
        printf("\n");
        printf(" -text                              - Default =      - Character/Text Overlay. Use Quotes.  Ex. -c "
               "\"Text Overlay\"\n");
        printf(
            " -textx                             - Default = 15   - Text Placement Horizontal from LEFT in Pixels\n");
        printf(" -texty = Text Y                    - Default = 25   - Text Placement Vertical from TOP in Pixels\n");
        printf(" -fontname = Font Name              - Default = 0    - Font Types (0-7), Ex. 0 = simplex, 4 = triplex, "
               "7 = script\n");
        printf(" -fontcolor = Font Color            - Default = 255 0 0  - Text blue (BGR)\n");
        printf(" -smallfontcolor = Small Font Color - Default = 0 0 255  - Text red (BGR)\n");
        printf(" -fonttype = Font Type              - Default = 0    - Font Line Type,(0-2), 0 = AA, 1 = 8, 2 = 4\n");
        printf(" -fontsize                          - Default = 0.5  - Text Font Size\n");
        printf(" -fontline                          - Default = 1    - Text Font Line Thickness\n");
        //printf(" -bgc = BG Color                    - Default =      - Text Background Color in Hex. 00ff00 = Green\n");
        //printf(" -bga = BG Alpha                    - Default =      - Text Background Color Alpha/Transparency 0-100\n");
        printf("\n");
        printf("\n");
        printf(" -latitude                          - Default = 60.7N (Whitehorse)   - Latitude of the camera.\n");
        printf(" -longitude                         - Default = 135.05W (Whitehorse) - Longitude of the camera\n");
        printf(" -angle                             - Default = -6 - Angle of the sun below the horizon. -6=civil "
               "twilight, -12=nautical twilight, -18=astronomical twilight\n");
        printf("\n");
        printf(" -preview                           - set to 1 to preview the captured images. Only works with a "
               "Desktop Environment \n");
        printf(" -time                              - Adds the time to the image. Combine with Text X and Text Y for "
               "placement \n");
        printf(" -darkframe                         - Set to 1 to disable time and text overlay \n");
        printf(" -showDetails                       - Set to 1 to display the metadata on the image \n");

        printf("%sUsage:\n", KRED);
        printf(" ./capture -width 640 -height 480 -exposure 5000000 -gamma 50 -type 1 -bin 1 -filename "
               "Lake-Laberge.PNG\n\n");
    }
    printf("%s", KNRM);

    const char *ext = strrchr(fileName, '.');
    if (strcmp(ext + 1, "jpg") == 0 || strcmp(ext + 1, "JPG") == 0 || strcmp(ext + 1, "jpeg") == 0 ||
        strcmp(ext + 1, "JPEG") == 0)
    {
        compression_parameters.push_back(CV_IMWRITE_JPEG_QUALITY);
        if (quality == 200)
        {
            quality = 95;
        }
    }
    else
    {
        compression_parameters.push_back(CV_IMWRITE_PNG_COMPRESSION);
        if (quality == 200)
        {
            quality = 3;
        }
    }
    compression_parameters.push_back(quality);

    int numDevices = ASIGetNumOfConnectedCameras();
    if (numDevices <= 0)
    {
        a_logger("\nNo Connected Camera...\n");
        width  = 1; //Set to 1 when NO Cameras are connected to avoid error: OpenCV Error: Insufficient memory
        height = 1; //Set to 1 when NO Cameras are connected to avoid error: OpenCV Error: Insufficient memory
    }
    else
    {
        a_logger("\nListing Attached Cameras:\n");
    }

    ASI_CAMERA_INFO ASICameraInfo;

    for (i = 0; i < numDevices; i++)
    {
        ASIGetCameraProperty(&ASICameraInfo, i);
        a_logger("- %d %s\n", i, ASICameraInfo.Name);        
    }

    if (ASIOpenCamera(CamNum) != ASI_SUCCESS)
    {
        a_logger("Open Camera ERROR, Check that you have root permissions!\n");
    }

    a_logger("\n%s Information:\n", ASICameraInfo.Name);    
    int iMaxWidth, iMaxHeight;
    double pixelSize;
    iMaxWidth  = ASICameraInfo.MaxWidth;
    iMaxHeight = ASICameraInfo.MaxHeight;
    pixelSize  = ASICameraInfo.PixelSize;
    a_logger("- Resolution:%dx%d\n", iMaxWidth, iMaxHeight);    
    a_logger("- Pixel Size: %1.1fμm\n", pixelSize);    
    a_logger("- Supported Bin: ");
    for (int i = 0; i < 16; ++i)
    {
        if (ASICameraInfo.SupportedBins[i] == 0)
        {
            break;
        }
        a_logger("%d ", ASICameraInfo.SupportedBins[i]);        
    }
    a_logger("\n");

    if (ASICameraInfo.IsColorCam)
    {
        a_logger("- Color Camera: bayer pattern:%s\n", bayer[ASICameraInfo.BayerPattern]);        
    }
    else
    {
        a_logger("- Mono camera\n");
    }
    if (ASICameraInfo.IsCoolerCam)
    {
        a_logger("- Camera with cooling capabilities\n");
    }

    const char *ver = ASIGetSDKVersion();
    a_logger("- SDK version %s\n", ver);

    if (ASIInitCamera(CamNum) == ASI_SUCCESS)
    {
        a_logger("- Initialise Camera OK\n");
    }
    else
    {
        a_logger("- Initialise Camera ERROR\n");
    }

    ASI_CONTROL_CAPS ControlCaps;
    int iNumOfCtrl = 0;
    ASIGetNumOfControls(CamNum, &iNumOfCtrl);
    for (i = 0; i < iNumOfCtrl; i++)
    {
        ASIGetControlCaps(CamNum, i, &ControlCaps);
        //a_logger("- %s\n", ControlCaps.Name);
    }

    if (width == 0 || height == 0)
    {
        width  = iMaxWidth;
        height = iMaxHeight;
    }

    long ltemp     = 0;
    ASI_BOOL bAuto = ASI_FALSE;
    ASIGetControlValue(CamNum, ASI_TEMPERATURE, &ltemp, &bAuto);
    a_logger("- Sensor temperature:%02f\n", (float)ltemp / 10.0);

    // Adjusting variables for chosen binning
    height    = height / bin;
    width     = width / bin;
    iTextX    = iTextX / bin;
    iTextY    = iTextY / bin;
    fontsize  = fontsize / bin;
    linewidth = linewidth / bin;

    const char *sType;
    if (Image_type == ASI_IMG_RAW16)
    {
        sType = "ASI_IMG_RAW16";
        pRgb.create(cvSize(width, height), CV_16UC1);
    }
    else if (Image_type == ASI_IMG_RGB24)
    {
        sType = "ASI_IMG_RGB24";
        pRgb.create(cvSize(width, height), CV_8UC3);
    }
    else
    {
        sType = "ASI_IMG_RAW8";
        pRgb.create(cvSize(width, height), CV_8UC1);
    }

    if (Image_type != ASI_IMG_RGB24 && Image_type != ASI_IMG_RAW16)
    {
        iStrLen     = strlen(buf);
        CvRect rect = cvRect(iTextX, iTextY - 15, iStrLen * 11, 20);
        cv::Mat roi = pRgb(rect);
        roi.setTo(cv::Scalar(180, 180, 180));
    }

    //-------------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------------------

    a_logger("%s", KGRN);
    a_logger("\nCapture Settings: \n");
    a_logger(" Image Type: %s\n", sType);
    a_logger(" Resolution: %dx%d \n", width, height);
    a_logger(" Quality: %d \n", quality);
    a_logger(" Exposure: %1.0fms\n", round(asiExposure / 1000));
    a_logger(" Max Exposure: %dms\n", asiMaxExposure);
    a_logger(" Auto Exposure: %d\n", asiAutoExposure);
    a_logger(" Gain: %d\n", asiGain);
    a_logger(" Max Gain: %d\n", asiMaxGain);
    a_logger(" Cooler Enabled: %d\n", asiCoolerEnabled);
    a_logger(" Target Temperature: %ldC\n", asiTargetTemp);
    a_logger(" Auto Gain: %d\n", asiAutoGain);
    a_logger(" Brightness: %d\n", asiBrightness);
    a_logger(" Gamma: %d\n", asiGamma);
    a_logger(" WB Red: %d\n", asiWBR);
    a_logger(" WB Blue: %d\n", asiWBB);
    a_logger(" Binning: %d\n", bin);
    a_logger(" Delay: %dms\n", delay);
    a_logger(" Daytime Delay: %dms\n", daytimeDelay);
    a_logger(" USB Speed: %d\n", asiBandwidth);
    a_logger(" Text Overlay: %s\n", ImgText);
    a_logger(" Text Position: %dpx left, %dpx top\n", iTextX, iTextY);
    a_logger(" Font Name:  %d\n", fontname[fontnumber]);
    a_logger(" Font Color: %d , %d, %d\n", fontcolor[0], fontcolor[1], fontcolor[2]);
    a_logger(" Small Font Color: %d , %d, %d\n", smallFontcolor[0], smallFontcolor[1], smallFontcolor[2]);
    a_logger(" Font Line Type: %d\n", linetype[linenumber]);
    a_logger(" Font Size: %1.1f\n", fontsize);
    a_logger(" Font Line: %d\n", linewidth);
    a_logger(" Outline Font : %d\n", outlinefont);
    a_logger(" Flip Image: %d\n", asiFlip);
    a_logger(" Filename: %s\n", fileName);
    a_logger(" Latitude: %s\n", latitude);
    a_logger(" Longitude: %s\n", longitude);
    a_logger(" Sun Elevation: %s\n", angle);
    a_logger(" Preview: %d\n", preview);
    a_logger(" Time: %d\n", time);
    a_logger(" Darkframe: %d\n", darkframe);
    a_logger(" Show Details: %d\n", showDetails);
    a_logger("%s", KNRM);

    ASISetROIFormat(CamNum, width, height, bin, (ASI_IMG_TYPE)Image_type);

    //-------------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------------------
    ASISetControlValue(CamNum, ASI_TEMPERATURE, 50 * 1000, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_BANDWIDTHOVERLOAD, asiBandwidth, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_EXPOSURE, asiExposure, asiAutoExposure == 1 ? ASI_TRUE : ASI_FALSE);
    ASISetControlValue(CamNum, ASI_AUTO_MAX_EXP, asiMaxExposure, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_GAIN, asiGain, asiAutoGain == 1 ? ASI_TRUE : ASI_FALSE);
    ASISetControlValue(CamNum, ASI_AUTO_MAX_GAIN, asiMaxGain, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_WB_R, asiWBR, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_WB_B, asiWBB, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_GAMMA, asiGamma, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_BRIGHTNESS, asiBrightness, ASI_FALSE);
    ASISetControlValue(CamNum, ASI_FLIP, asiFlip, ASI_FALSE);
    if (ASICameraInfo.IsCoolerCam)
    {
        ASI_ERROR_CODE err = ASISetControlValue(CamNum, ASI_COOLER_ON, asiCoolerEnabled == 1 ? ASI_TRUE : ASI_FALSE, ASI_FALSE);
	if (err != ASI_SUCCESS)
	{
		a_logger("%s", KRED);
		a_logger(" Could not enable cooler\n");
		a_logger("%s", KNRM);
	}
	err = ASISetControlValue(CamNum, ASI_TARGET_TEMP, asiTargetTemp, ASI_FALSE);
	if (err != ASI_SUCCESS)
        {
                a_logger("%s", KRED);
                a_logger(" Could not set cooler temperature\n");
                a_logger("%s", KNRM);
        }
    }

    pthread_t thread_display = 0;
    if (preview == 1)
    {
        bDisplay = 1;
        pthread_create(&thread_display, NULL, Display, (void *)&pRgb);
    }

    if (!bSaveRun)
    {
        bSaveRun = true;
        if (pthread_create(&hthdSave, 0, SaveImgThd, 0) != 0)
        {
            bSaveRun = false;
        }
    }

    // Initialization
    int currentExposure = asiExposure;
    int exp_ms          = 0;
    long autoGain       = 0;
    long autoExp        = 0;
    int useDelay        = 0;

    while (bMain)
    {
        bool needCapture = true;
        std::string lastDayOrNight;
        int captureTimeout = -1;

        // Find out if it is currently DAY or NIGHT
        calculateDayOrNight(latitude, longitude, angle);

        lastDayOrNight = dayOrNight;
        a_logger("\n");
        if (dayOrNight == "DAY")
        {
            // Setup the daytime capture parameters
            if (endOfNight == true)
            {
                system("scripts/endOfNight.sh &");
                endOfNight = false;
            }
            if (daytimeCapture != 1)
            {
                needCapture = false;
                a_logger("It's daytime... we're not saving images\n");
                usleep(daytimeDelay * 1000);
            }
            else
            {
                a_logger("Starting daytime capture\n");
                a_logger("Saving auto exposed images every %d ms\n\n", daytimeDelay);
                exp_ms         = 32;
                useDelay       = daytimeDelay;
                captureTimeout = exp_ms <= 100 ? 200 : exp_ms * 2;
                ASISetControlValue(CamNum, ASI_EXPOSURE, exp_ms, ASI_TRUE);
                ASISetControlValue(CamNum, ASI_GAIN, 0, ASI_FALSE);
            }
        }
        else if (dayOrNight == "NIGHT")
        {
            // Setup the night time capture parameters
            if (asiAutoExposure == 1)
            {
                a_logger("Saving auto exposed images every %d ms\n\n", delay);
            }
            else
            {
                a_logger("Saving %ds exposure images every %d ms\n\n", (int)round(currentExposure / 1000000), delay);
            }
            // Set exposure value for night time capture
            useDelay = delay;
            ASISetControlValue(CamNum, ASI_EXPOSURE, currentExposure, asiAutoExposure == 1 ? ASI_TRUE : ASI_FALSE);
            ASISetControlValue(CamNum, ASI_GAIN, asiGain, asiAutoGain == 1 ? ASI_TRUE : ASI_FALSE);
        }
        a_logger("Press Ctrl+C to stop\n\n");

        /*
        William - Not sure why we are ignoring auto exposure changes

        long lastExp = 0;
        int flushingStatus = 0;
        */
        if (needCapture)
        {
            ASIStartVideoCapture(CamNum);
            while (bMain && lastDayOrNight == dayOrNight)
            {
                if (ASIGetVideoData(CamNum, pRgb.data, pRgb.step[0] * pRgb.rows, captureTimeout) == ASI_SUCCESS)
                {
                    // Read current camera parameters
                    ASIGetControlValue(CamNum, ASI_EXPOSURE, &autoExp, &bAuto);
                    ASIGetControlValue(CamNum, ASI_GAIN, &autoGain, &bAuto);
                    ASIGetControlValue(CamNum, ASI_TEMPERATURE, &ltemp, &bAuto);

                    /*
                    William - Not sure why we are ignoring auto exposure changes

                    if (lastExp != autoExp)
                    {
                        flushingStatus = 2;
                        a_logger("exp changed %ld -> %ld, flushing ...\n", lastExp, autoExp);
                    }
                    lastExp = autoExp;
                    if (flushingStatus != 0)
                    {
                        a_logger("flushing %d\n", flushingStatus);
                        flushingStatus --;
                        continue;
                    }
                    */
                    if (brokenDetector(pRgb))
                    {
                        a_logger("bad image detected!\n");
                        continue;
                    }

                    /**
                    The following function will check if the image is a "mosaic", meaning it is composed
                    of multiple rectangles of various images. This happens on some ASI120MC clone cameras                    
                    **/
                    /*if (mosaicImage(pRgb))
                    {
                        a_logger("Mosaic Image deteced! Trying again\n");
                        continue;
                    }*/

                    // Get Current Time for overlay
                    sprintf(bufTime, "%s", getTime());

                    if (darkframe != 1)
                    {
                        // If darkframe mode is off, add overlay text to the image
                        int iYOffset = 0;
                        //cvText(pRgb, ImgText, iTextX, iTextY+(iYOffset/bin), fontsize, linewidth, linetype[linenumber], fontname[fontnumber], fontcolor, Image_type);
                        //iYOffset+=30;
                        if (time == 1)
                        {
                            cvText(pRgb, bufTime, iTextX, iTextY + (iYOffset / bin), fontsize, linewidth,
                                   linetype[linenumber], fontname[fontnumber], fontcolor, Image_type, outlinefont);
                            iYOffset += 30;
                        }

                        if (showDetails == 1)
                        {
                            sprintf(bufTemp, "Sensor %.1fC", (float)ltemp / 10);
                            cvText(pRgb, bufTemp, iTextX, iTextY + (iYOffset / bin), fontsize * 0.8, linewidth,
                                   linetype[linenumber], fontname[fontnumber], smallFontcolor, Image_type, outlinefont);
                            iYOffset += 30;
                            sprintf(bufTemp, "Exposure %.3f s", (float)autoExp / 1000000);
                            cvText(pRgb, bufTemp, iTextX, iTextY + (iYOffset / bin), fontsize * 0.8, linewidth,
                                   linetype[linenumber], fontname[fontnumber], smallFontcolor, Image_type, outlinefont);
                            iYOffset += 30;
                            sprintf(bufTemp, "Gain %d", (int)autoGain);
                            cvText(pRgb, bufTemp, iTextX, iTextY + (iYOffset / bin), fontsize * 0.8, linewidth,
                                   linetype[linenumber], fontname[fontnumber], smallFontcolor, Image_type, outlinefont);
                            iYOffset += 30;
                        }
                    }
                    a_logger("Exposure value: %.0f µs\n", (float)autoExp);
                    if (asiAutoExposure == 1)
                    {
                        // Retrieve the current Exposure for smooth transition to night time
                        // as long as auto-exposure is enabled during night time
                        currentExposure = autoExp;
                    }

                    // Save the image
                    a_logger("Saving...");
                    a_logger(bufTime);
                    a_logger("\n");
                    if (!bSavingImg)
                    {
                        pthread_mutex_lock(&mtx_SaveImg);
                        saveExpTime = autoExp;
                        pthread_cond_signal(&cond_SatrtSave);
                        pthread_mutex_unlock(&mtx_SaveImg);
                    }

                    if (asiAutoGain == 1 && dayOrNight == "NIGHT")
                    {
                        ASIGetControlValue(CamNum, ASI_GAIN, &autoGain, &bAuto);
                        a_logger("Auto Gain value: %d\n", (int)autoGain);
                        writeToLog(autoGain);
                    }

                    if (asiAutoExposure == 1)
                    {
                        a_logger("Auto Exposure value: %d ms\n", (int)round(autoExp / 1000));
                        writeToLog(autoExp);
                        if (dayOrNight == "NIGHT")
                        {
                            ASIGetControlValue(CamNum, ASI_EXPOSURE, &autoExp, &bAuto);
                        }
                        else
                        {
                            currentExposure = autoExp;
                        }

                        // Delay applied before next exposure
                        if (autoExp < asiMaxExposure * 1000 && dayOrNight == "NIGHT")
                        {
                            // if using auto-exposure and the actual exposure is less than the max,
                            // we still wait until we reach maxexposure. This is important for a
                            // constant frame rate during timelapse generation
                            a_logger("Sleeping: %d ms\n", asiMaxExposure - (int)(autoExp / 1000) + useDelay);
                            usleep((asiMaxExposure * 1000 - autoExp) + useDelay * 1000);
                        }
                        else
                        {
                            usleep(useDelay * 1000);
                        }
                    }
                    else
                    {
                        usleep(useDelay * 1000);
                    }
                    calculateDayOrNight(latitude, longitude, angle);
                }
            }
            if (lastDayOrNight == "NIGHT")
            {
                endOfNight = true;
            }
            ASIStopVideoCapture(CamNum);
        }
    }
    ASICloseCamera(CamNum);

    if (bDisplay)
    {
        bDisplay = 0;
        pthread_join(thread_display, &retval);
    }

    if (bSaveRun)
    {
        bSaveRun = false;
        pthread_mutex_lock(&mtx_SaveImg);
        pthread_cond_signal(&cond_SatrtSave);
        pthread_mutex_unlock(&mtx_SaveImg);
        pthread_join(hthdSave, 0);
    }
    a_logger("main function over\n");
    return 1;
}
