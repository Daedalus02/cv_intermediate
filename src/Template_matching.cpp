#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath> // Per std::abs

namespace fs = std::filesystem;

// Funzione per ruotare un'immagine
cv::Mat rotateImage(const cv::Mat& source, double angle) {
    cv::Point2f center(source.cols / 2.0f, source.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat result;
    cv::warpAffine(source, result, rotMat, source.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return result;
}

int main() {
    std::string immagineTestPath = "../data/004_sugar_box/test_images/4_0049_000003-color.jpg";
    std::string modelsFolderPath = "../data/004_sugar_box/models/";
    double confidenceThreshold = 0.4;
    std::vector<double> scales = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> angles = {-15, -10, -5, 0, 5, 10, 15}; // Prova diverse rotazioni (in gradi)

    cv::Mat imgTestColor = cv::imread(immagineTestPath, cv::IMREAD_COLOR);
    cv::Mat imgTestGray = cv::imread(immagineTestPath, cv::IMREAD_GRAYSCALE);

    if (imgTestColor.empty() || imgTestGray.empty()) {
        std::cerr << "Errore nel caricamento dell'immagine di test." << std::endl;
        return -1;
    }

    double maxValGlobal = -1;
    cv::Point maxLocGlobal;
    cv::Size templateSizeGlobal;

    for (const auto& entry : fs::directory_iterator(modelsFolderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            std::string templatePath = entry.path().string();
            cv::Mat templGrayOriginal = cv::imread(templatePath, cv::IMREAD_GRAYSCALE);

            if (!templGrayOriginal.empty()) {
                for (double scale : scales) {
                    cv::Mat templGrayResized;
                    cv::resize(templGrayOriginal, templGrayResized, cv::Size(), scale, scale);

                    for (double angle : angles) {
                        cv::Mat templGrayRotated = rotateImage(templGrayResized, angle);

                        if (imgTestGray.cols >= templGrayRotated.cols && imgTestGray.rows >= templGrayRotated.rows) {
                            cv::Mat result;
                            cv::matchTemplate(imgTestGray, templGrayRotated, result, cv::TM_CCOEFF_NORMED);
                            double minVal; double maxVal;
                            cv::Point minLoc; cv::Point maxLoc;
                            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                            if (maxVal > maxValGlobal) {
                                maxValGlobal = maxVal;
                                maxLocGlobal = cv::Point(cvRound(maxLoc.x), cvRound(maxLoc.y));
                                templateSizeGlobal = cv::Size(templGrayRotated.cols, templGrayRotated.rows);
                            }
                        }
                    }
                }
            }
        }
    }

    if (maxValGlobal > confidenceThreshold) {
        cv::Point topLeft = maxLocGlobal;
        cv::Point bottomRight(topLeft.x + templateSizeGlobal.width, topLeft.y + templateSizeGlobal.height);
        cv::rectangle(imgTestColor, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);
        std::cout << "Oggetto trovato con confidenza: " << maxValGlobal << std::endl;
    } else {
        std::cout << "Nessuna corrispondenza sufficientemente buona trovata con nessun template (soglia: " << confidenceThreshold << "). Max confidenza: " << maxValGlobal << std::endl;
    }

    cv::imshow("Immagine Test con Rilevamento", imgTestColor);
    cv::waitKey(0);

    return 0;
}