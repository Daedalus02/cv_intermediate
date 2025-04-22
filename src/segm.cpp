
    

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void segmentaOggetti(Mat& image) {
    // Converti in scala di grigi
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Applica blur per ridurre il rumore
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // Sogliatura binaria (Otsu)
    Mat binary;
    threshold(blurred, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    // Operazioni morfologiche per pulire l'immagine
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    // Trova i contorni
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Disegna i contorni e numeri gli oggetti
    RNG rng(12345);
    Mat result = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        // Ignora contorni troppo piccoli (rumore)
        if (contourArea(contours[i]) < 500)
            continue;

        // Colore casuale per ogni oggetto
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // Disegna il contorno
        drawContours(result, contours, (int)i, color, 2);

        // Disegna il numero dell'oggetto
        Moments m = moments(contours[i]);
        Point center(m.m10 / m.m00, m.m01 / m.m00);
        putText(result, to_string(i + 1), center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    }

    // Mostra i risultati
    imshow("Immagine originale", image);
    imshow("Maschera binaria", binary);
    imshow("Oggetti segmentati", result);
    waitKey(0);
}

int main(int argc, char** argv) {
    // Carica l'immagine
    Mat image = imread("../../data/004_sugar_box/test_images/4_0001_000121-color.jpg");

    if (image.empty()) {
        cout << "Impossibile caricare l'immagine" << endl;
        return -1;
    }

    // Ridimensiona se troppo grande
    if (image.rows > 800 || image.cols > 800) {
        resize(image, image, Size(800, 800 * image.rows / image.cols));
    }

    segmentaOggetti(image);
    destroyAllWindows();
    return 0;
}