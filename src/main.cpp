/**
 * References::
 * https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
 * https://www.youtube.com/watch?v=HMGX8HXskKk
 */
#include <opencv2/opencv.hpp>
#include <queue>


using namespace cv;
using namespace std;

class NodePixel;

class Edge{
public:
    int toRowIndex;
    int toColIndex;
    double capacity;
    Edge(){
        capacity =0;
    };

    Edge(int toRowIndex, int toColIndex, double capacity) : toRowIndex(toRowIndex), toColIndex(toColIndex),
                                                            capacity(capacity) {}

    bool isToSink() {
        if (toRowIndex==-2 && toColIndex==-2)
            return true;
        return false;
    }
    bool isToSource() {
        if (toRowIndex==-1 && toColIndex==-1)
            return true;
        return false;
    }
};

class NodePixel {
public:
    int rowIndex;
    int columnIndex;
    int parentRowIndex;
    int parentColIndex;
    bool isTraversed;
    vector< Edge > edgeList;
    NodePixel(){
        rowIndex =0;
        columnIndex =0;
        isTraversed=false;
    };
    NodePixel(int rowIndex, int columnIndex) : rowIndex(rowIndex), columnIndex(columnIndex) {}

    void setParent(int rowIndex,int colIndex){
        parentRowIndex = rowIndex;
        parentColIndex = colIndex;
    };
    void addEdge(int toRowIndex, int toColIndex,float weight){
        edgeList.push_back(Edge(toRowIndex, toColIndex,weight));
    };

    Edge & getEdge(int rowIndex,int columnIndex){
        for(int i=0;i<edgeList.size();i++){
            Edge edge = edgeList.at(i);
            if(edge.toRowIndex==rowIndex && edge.toColIndex==columnIndex)
                return edgeList.at(i);
        }
    };

    bool isSink() {
        if (rowIndex==-2 && columnIndex==-2)
            return true;
        return false;
    }
    bool isSource() {
        if (rowIndex==-1 && columnIndex==-1)
            return true;
        return false;
    }
};

class PixelIndex{
public:
    int i;
    int j;
    PixelIndex(int indexi,int indexj){
        i=indexi;
        j=indexj;
    }
};


float fordFulkerson(vector< vector< NodePixel > > &adjacencyList,NodePixel &superSource,NodePixel &superSink,int rows,int cols,Mat &out_image);
bool bfs(NodePixel pixel, vector< vector< NodePixel > > &adjacencyList,NodePixel &sinkNode);
double getCapacity(Edge edge, double maxEdgeVal);

int main(int argc, char **argv) {
//namedWindow("Original image", WINDOW_AUTOSIZE);
    if (argc != 4) {
        cout << "Usage: ../seg input_image initialization_file output_mask" << endl;
        return -1;
    }
    // Load the input image
    // the image should be a 3 channel image by default but we will double check that in teh seam_carving
    Mat in_image;
    in_image = imread(argv[1]/*, CV_LOAD_IMAGE_COLOR*/);

    if (!in_image.data) {
        cout << "Could not load input image!!!" << endl;
        return -1;
    }

    if (in_image.channels() != 3) {
        cout << "Image does not have 3 channels!!! " << in_image.depth() << endl;
        return -1;
    }

    // the output image
    Mat out_image = in_image.clone();

    ifstream f(argv[2]);
    if (!f) {
        cout << "Could not load initial mask file!!!" << endl;
        return -1;
    }

    int width = in_image.cols;
    int height = in_image.rows;

    int n;
    f >> n;

    Mat gaussianOutput;
    Mat greyScale;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;


    //smoothening output
    GaussianBlur( in_image, gaussianOutput, Size(3,3), 0, 0, BORDER_DEFAULT );

    cvtColor( gaussianOutput, gaussianOutput, CV_BGR2GRAY);

    Mat input_gray = gaussianOutput.clone();

    normalize(gaussianOutput, gaussianOutput, 0, 255, NORM_MINMAX, CV_8UC1);

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Scharr( input_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    //Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    Scharr( input_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    //Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    //average gradient
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, greyScale );

    /* Mat greyScale;
     cvtColor(in_image, greyScale, CV_BGR2GRAY);

    */
    NodePixel superSource(-1,-1);
    NodePixel superSink(-2,-2);

    std::vector< std::vector<NodePixel> > adjacencyList(greyScale.rows, vector<NodePixel>(greyScale.cols,NodePixel()));

    for(int i=0;i<greyScale.rows;i++){
        for(int j=0;j<greyScale.cols;j++){
            adjacencyList.at(i).at(j).rowIndex =i;
            adjacencyList.at(i).at(j).columnIndex =j;
        }
    }

    float maxFlow=0;
    float sumForSink = 0;
    int numSink = 0;
    float sumForSource = 0;
    float meanSink = 0;
    int numSource = 0;
    float meanSource = 0;
    // get the initil pixels
    for (int i = 0; i < n; ++i) {
        int x, y, t;
        f >> x >> y >> t;

        if (x < 0 || x >= width || y < 0 || y >= height) {
            cout << "I valid pixel mask!" << endl;

            return -1;
        }
        if (t == 0) {
            NodePixel &nodePixel = adjacencyList.at(y).at(x);
            nodePixel.addEdge(-2,-2,LONG_MAX);
            sumForSink += greyScale.at<float>(y, x);
            numSink++;

        } else {
            superSource.addEdge(y,x,LONG_MAX);
            sumForSource += greyScale.at<float>(y, x);
            numSource++;
        }
    }

    double maxEdgeVal=0;
    for (int i = 0; i < greyScale.rows; i++) {
        for (int j = 0; j < greyScale.cols; j++) {
            double edge_weight = 0;
            NodePixel &nodePixel=adjacencyList.at(i).at(j);

            vector<PixelIndex> pixelList;
            if(i>0){
                PixelIndex pixelIndex(i-1,j);
                pixelList.push_back(pixelIndex);
            }
            if(i<height-1){
                PixelIndex pixelIndex(i+1,j);
                pixelList.push_back(pixelIndex);
            }
            if(j>0){
                PixelIndex pixelIndex(i,j-1);
                pixelList.push_back(pixelIndex);
            }
            if(j<width-1){
                PixelIndex pixelIndex(i,j+1);
                pixelList.push_back(pixelIndex);
            }
            for(int pixelIndex=0;pixelIndex<pixelList.size();pixelIndex++){
                double diff = (gaussianOutput.at<uchar>(i,j) - gaussianOutput.at<uchar>(pixelList.at(pixelIndex).i, pixelList.at(pixelIndex).j));
                if(diff<0.5){
                    edge_weight = LONG_MAX;
                }else{
                    edge_weight = 1;
                }
                nodePixel.addEdge(pixelList.at(pixelIndex).i, pixelList.at(pixelIndex).j, edge_weight);
            }
        }
    }
    maxFlow = fordFulkerson(adjacencyList,superSource,superSink,greyScale.rows,greyScale.cols,out_image);

    // write it on disk
    imwrite(argv[3], out_image);

    // also display them both

    namedWindow("Original image", WINDOW_AUTOSIZE);
    namedWindow("Show Marked Pixels", WINDOW_AUTOSIZE);
    imshow("Original image", in_image);
    imshow("Show Marked Pixels", out_image);
    waitKey(0);
    return 0;
};

float fordFulkerson(vector< vector< NodePixel > > &adjacencyList,NodePixel &superSource,NodePixel &superSink,int rows,int cols,Mat &out_image)
{
    float maxFlow = 0;
    int numOfPath=0;
    while(bfs(superSource, adjacencyList, superSink)) {
        numOfPath++;
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                superSink.isTraversed   = false;
                adjacencyList.at(i).at(j).isTraversed= false;
            }
        }
        NodePixel traversalNode = adjacencyList.at(superSink.parentRowIndex).at(superSink.parentColIndex);
        double minFlow = LONG_MAX;
//        cout<<"::::::::::::::::NEW PATH::::::::::::"<<endl;
        while (!traversalNode.isSource()) {
            NodePixel parentPixel;
            if (traversalNode.parentColIndex == -1 && traversalNode.parentRowIndex == -1)
                parentPixel = superSource;
            else
                parentPixel = adjacencyList.at(traversalNode.parentRowIndex).at(traversalNode.parentColIndex);
//            cout<<"("<<traversalNode.parentRowIndex<<"  "<<traversalNode.parentColIndex<<") :to: "<<"("<<traversalNode.rowIndex<<"  "<<traversalNode.columnIndex<<")" <<" Flow:"<<parentPixel.getEdge(traversalNode.rowIndex, traversalNode.columnIndex).capacity<<endl;
            minFlow = min(minFlow, parentPixel.getEdge(traversalNode.rowIndex, traversalNode.columnIndex).capacity);
            traversalNode = parentPixel;
        }

        traversalNode = adjacencyList.at(superSink.parentRowIndex).at(superSink.parentColIndex);

        while (true) {
            if (traversalNode.parentColIndex == -1 && traversalNode.parentRowIndex == -1)
                break;

            NodePixel copyParentPixel = adjacencyList.at(traversalNode.parentRowIndex).at(traversalNode.parentColIndex);
            NodePixel &parentPixel = adjacencyList.at(traversalNode.parentRowIndex).at(traversalNode.parentColIndex);
            Edge &fromEdge = parentPixel.getEdge(traversalNode.rowIndex, traversalNode.columnIndex);
            Edge &toEdge = traversalNode.getEdge(parentPixel.rowIndex, parentPixel.columnIndex);
            fromEdge.capacity -= minFlow;
            toEdge.capacity += minFlow;

            traversalNode = copyParentPixel;
        }
        maxFlow += minFlow;
//        cout << endl << "MINFLOW::" << minFlow << endl;
    }
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            Vec3b pixel = out_image.at<Vec3b>(i, j);
            if(adjacencyList.at(i).at(j).isTraversed){
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            }else{
                pixel[0] = 255;
                pixel[1] = 0;
                pixel[2] = 0;
            }
            out_image.at<Vec3b>(i, j) = pixel;
        }
    }
//    cout<<endl<<"NumOfPath"<<numOfPath<<endl;
    return maxFlow;
}

bool bfs(NodePixel pixel, vector< vector< NodePixel > > &adjacencyList,NodePixel &sinkNode){
    queue < NodePixel > q;
    queue < NodePixel > empty;
    q.push(pixel);
    while (!q.empty())
    {
        NodePixel u = q.front();
        q.pop();
        for (int v=0; v<u.edgeList.size(); v++)
        {
            Edge edge = u.edgeList.at(v);
            if(edge.toRowIndex>=0 && edge.toColIndex>=0){
                NodePixel &nodePixel = adjacencyList.at(edge.toRowIndex).at(edge.toColIndex);
                if (!nodePixel.isTraversed && edge.capacity > 0)
                {
                    nodePixel.isTraversed = true;
                    nodePixel.setParent(u.rowIndex,u.columnIndex);
                    q.push(nodePixel);
                }
            }else if(edge.isToSink() && edge.capacity>0){
                sinkNode.isTraversed =true;
                sinkNode.setParent(u.rowIndex,u.columnIndex);
                swap(q,empty);
                return true;
            }
        }
    }
    return sinkNode.isTraversed;
}
