#include "../include/performance_metrics.h"


const std::array<const char *, 3> class_names = {"Sugar", "Mustard", "Power Drill"};

// FUNCTION MEMBERS
void PerformanceMetrics:: compute_IoU(){

     // Parser for predicted labels
    parser(this->path_pred_labels, this->sugar_p, this->mustard_p, this->power_drill_p);
    // Parser for true labels
    parser(this->path_true_labels, this->sugar_t, this->mustard_t, this->power_drill_t);

    std::cout<<"Predicted labels:\n\n"; // SOLO PER DEBUG
    print_value(this->sugar_p, this->mustard_p, this->power_drill_p);

    std::cout<<"\nTrue labels:\n\n"; // SOLO PER DEBUG
    print_value(this->sugar_t, this->mustard_t, this->power_drill_t);
    std::cout<<"\n";

    // Initialize 2 vectors for the rectangle of our Algo dectector prediction, and for the true label of the dataset
    // In each cell of each vector we find the coordinates of the top left and bottom right corners
    std::vector<cv::Rect> rect_p(3) , rect_t(3);
    rect_p[0] = cv::Rect(sugar_p[0], sugar_p[1]);

    if(sugar_t[0].x == 0 && sugar_t[0].y == 0 && sugar_t[1].x == 0 && sugar_t[1].y == 0) this->miss[0] = true;
    rect_t[0] = cv::Rect(sugar_t[0], sugar_t[1]);

    rect_p[1] = cv::Rect(mustard_p[0], mustard_p[1]);
    if(mustard_t[0].x == 0 && mustard_t[0].y == 0 && mustard_t[1].x == 0 && mustard_t[1].y == 0) this->miss[1] = true;
    rect_t[1] = cv::Rect(mustard_t[0], mustard_t[1]);

    rect_p[2] = cv::Rect(power_drill_p[0], power_drill_p[1]);
    if(power_drill_t[0].x == 0 && power_drill_t[0].y == 0 && power_drill_t[1].x == 0 && power_drill_t[1].y == 0) this->miss[2] = true;
    rect_t[2] = cv::Rect(power_drill_t[0], power_drill_t[1]);

    // Compute and storage IoU array in the data member: IoU =  Area of overlap / Area of union

    // Define the variable to store the areas of intersection, union and the respective IoU
    double areas_int[3];
    double areas_union[3];

    for (int i = 0; i < rect_p.size(); ++i)
    {
        if (!miss[i])
        {
            cv::Rect intersect = rect_p[i] & rect_t[i];
            areas_int[i] = intersect.area();
            std::cout << "Intersection A (" << class_names[i] << "): " << areas_int[i] << std::endl; // QUesto serve solo in fase di debug

            areas_union[i] = rect_p[i].area() + rect_t[i].area() - areas_int[i];
            std::cout << "Union A (" << class_names[i] << "): " << areas_union[i] << std::endl; // QUesto serve solo in fase di debug

            this->IoU[i] = areas_int[i] / areas_union[i];
            std::cout << "IoU (" << class_names[i] << "): " << this->IoU[i] << "\n\n"; // QUesto serve solo in fase di debug
        }
    }
}

double PerformanceMetrics:: print_metrics(){

    this->compute_IoU();
   
    // Calculus of the metrics
    // - MIoU = (IoU1 +  IoU2 + IoU3) / 3
    // - Accuracy : if IoU[i] > 0.5 ==> TRUE POSITIVE

    // MIoU
    int m = 0;
    double sum = 0;
    for (int i = 0; i < std::size(this->IoU); i++)
    {
        if (!this->miss[i])
        {
            m++;
            sum += this->IoU[i];
        }
    }
    double MIoU = sum / m;
    std::cout << "MIoU " << MIoU << "\n\n";
    // STAMPA IN UN FILE

    // Accuracy 
    std::cout<<"Accuracy : \n";
    for (int i = 0; i < std::size(this->IoU); i++){
        if(!this->miss[i]){
            if(this->IoU[i] > 0.5){
                std::cout << "IoU for " << class_names[i] << " is true positive" << std::endl;
            }else{
                std::cout << "IoU for " << class_names[i] << " is NOT a true positive" << std::endl;
            }
        }
    }
    // STAMPA IN UN FILE

    return MIoU; // LA FUNZIONE SARA VOID
}

// HELPER FUCNTIONS
//std::vector<cv::Point>
 void parser(const std::string& path , std::vector<cv:: Point2f>& sugar, std::vector<cv:: Point2f>& mustard, std::vector<cv:: Point2f>& power_drill){

    std::ifstream file(path);

    if(!file.is_open()){
        std::cerr<< "Impossibile to open the file \n";
        return;
    }
    std::string line;

    while(std::getline(file,line)){

        std::vector<std::string> words;
        std::istringstream iss(line);
        std::string word;
        
        // Dividing the line using space as separator
        while(iss >> word) {
            words.push_back(word);
        }

        if(words[0] == "004_sugar_box"){

            sugar[0] = cv::Point2f(std::stoi(words[1]), std::stoi(words[2]));
            sugar[1] = cv::Point2f(std::stoi(words[3]), std::stoi(words[4]));

        }else if(words[0] == "006_mustard_bottle"){

            mustard[0] = cv::Point2f(std::stoi(words[1]), std::stoi(words[2]));
            mustard[1] = cv::Point2f(std::stoi(words[3]), std::stoi(words[4]));

        }else if (words[0] == "035_power_drill")
        {
            power_drill[0] = cv::Point2f(std::stoi(words[1]), std::stoi(words[2]));
            power_drill[1] = cv::Point2f(std::stoi(words[3]), std::stoi(words[4]));
        }
        
        // closed automatically the file open thanks to the distructor of ifstream
        
    }

}

void print_value(std::vector<cv::Point2f> v1, std::vector<cv::Point2f> v2, std::vector<cv::Point2f> v3) {

    auto print_coords = [](const std::string& label, const std::vector<cv::Point2f>& vec) {
        if (vec.size() == 2 && vec[0] == cv::Point2f(0.0f, 0.0f) && vec[1] == cv::Point2f(0.0f, 0.0f)) {
            std::cout << label << ": not present" << std::endl;
        } else {
            std::cout << label << ":" << std::endl;
            for (const auto& p : vec) {
                std::cout << "(" << p.x << ", " << p.y << ") ";
            }
            std::cout << std::endl;
        }
    };
    print_coords("Sugar Coord", v1);
    print_coords("Mustard Coord", v2);
    print_coords("Power Drill Coord", v3);
}

/*
 void printValue(std::vector<cv:: Point2f> v1, std::vector<cv:: Point2f> v2, std::vector<cv:: Point2f> v3){

    
    
    std::cout << "Sugar Coord:" << std::endl;
    for (const auto& p : v1) {
        std::cout << "(" << p.x << ", " << p.y << ") ";
    }
    std::cout << std::endl;

    std::cout << "Mustard Coord:" << std::endl;
    for (const auto& p : v2) {
        std::cout << "(" << p.x << ", " << p.y << ") ";
    }
    std::cout << std::endl;

    std::cout << "Power Drill Coord:" << std::endl;
    for (const auto& p : v3) {
        std::cout << "(" << p.x << ", " << p.y << ")";
    }
    std::cout << "\n\n";

}
*/