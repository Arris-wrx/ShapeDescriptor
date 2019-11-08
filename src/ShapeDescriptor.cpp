#include "ShapeDescriptor.h"

#include <iostream>
#include "fourier_descriptors.hpp"

using namespace std;
using namespace cv;

// flag tf = 1 for train and 0 for use
ShapeDescriptor::ShapeDescriptor(bool tf, const string data_f_path, const string data_path) : trainFlag(tf), path(data_path), fpath(data_f_path)
{
    if (trainFlag)
    {
        fs.open(path, FileStorage::READ);

        if (!fs.isOpened())
        {
            cerr << "Failed to open " << path << endl;
            exit;
        }
        // read Idents
        FileNode n = fs["Idents"];                         // Read string sequence - Get node
        if (n.type() != FileNode::SEQ)
        {
            cerr << "Idents is not a sequence! FAIL" << endl;
            exit;
        }
        std::cout << endl;
        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
        {
            idx.push_back((int)*it);
        }

        // read Descriptors
        FileNode m = fs["Contours"];                         // Read string sequence - Get node
        if (m.type() != FileNode::SEQ)
        {
            cerr << "Contours is not a sequence! FAIL" << endl;
            exit;
        }

        it = m.begin(), it_end = m.end(); // Go through the node
        for (; it != it_end; ++it)
        {
            Mat A;
            *it >> A;
            mat.push_back(A);
        }

        //[debug print]
        /*std::cout << endl;
        cout << "idx = " << "\n";
        for (auto a : idx)
            std::cout << a << "\n";

        cout << "Mat = " << "\n";
        for (auto a : mat)
            std::cout << a << "\n";*/
            //[\debug print]

        // calculate uniq indexes
        copy(idx.begin(), idx.end(), back_inserter(u_idx));
        sort(u_idx.begin(), u_idx.end());
        u_idx.erase(unique(u_idx.begin(), u_idx.end()), u_idx.end());

        //[debug print]
        /*cout << "u_idx = " << "\n";
        for (auto a : u_idx)
            std::cout << a << "\n";*/
        //[\debug print]

        // calculate average descriptor matrix
        for (int i = 0; i < u_idx.size(); ++i)
        {
            avgDesc.push_back(Mat::zeros(mat[0].rows, mat[0].cols, CV_64FC2));
        }
        vector<int> idx_cnt(u_idx.size(), 0);
        for (int j = 0; j < idx.size(); ++j)
        {
            int index = idx[j];
            int pos;
            for (int k = 0; k < u_idx.size(); ++k)
            {
                if (index == u_idx[k])
                {
                    pos = k;
                    break;
                }
            }
            idx_cnt[pos]++;
            avgDesc[pos] += mat[j];
        }

        for (int i = 0; i < avgDesc.size(); ++i)
            avgDesc[i] /= Scalar(idx_cnt[i], idx_cnt[i]);

        //[debug print]
        /*cout << "avgMat = " << "\n";
        for (auto a : avgDesc)
            std::cout << a << "\n";*/
        //[\debug print]

        fs.release();

    }
    else
    {
        fsFinal.open(fpath, FileStorage::READ);
        if (!fsFinal.isOpened())
        {
            cerr << "Failed to open " << fpath << endl;
            exit;
        }

        // read Idents
        FileNode n = fsFinal["Idents"];                         // Read string sequence - Get node
        if (n.type() != FileNode::SEQ)
        {
            cerr << "Idents is not a sequence! FAIL" << endl;
            exit;
        }
        std::cout << endl;
        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
        {
            u_idx.push_back((int)*it);
        }

        // read Descriptors
        FileNode m = fsFinal["AverageContours"];                         // Read string sequence - Get node
        if (m.type() != FileNode::SEQ)
        {
            cerr << "Contours is not a sequence! FAIL" << endl;
            exit;
        }

        it = m.begin(), it_end = m.end(); // Go through the node
        for (; it != it_end; ++it)
        {
            Mat A;
            *it >> A;
            avgDesc.push_back(A);
        }


        fsFinal.release();
    }


}

ShapeDescriptor::~ShapeDescriptor()
{
    if (trainFlag)
    {
        fs.open(path, FileStorage::WRITE);
        fsFinal.open(fpath, FileStorage::WRITE);

        fs << "Contours";                              // text - mapping
        fs << "[";

        for (int i = 0; i < mat.size(); ++i)
        {
            fs << mat[i];
        }

        fs << "]";

        fs << "Idents";                              // text - mapping
        fs << "[";

        for (int i = 0; i < idx.size(); ++i)
        {
            fs << idx[i];
        }

        fs << "]";

        //---

        fsFinal << "AverageContours";                              // text - mapping
        fsFinal << "[";

        for (int i = 0; i < avgDesc.size(); ++i)
        {
            fsFinal << avgDesc[i];
        }

        fsFinal << "]";

        fsFinal << "Idents";                              // text - mapping
        fsFinal << "[";

        for (int i = 0; i < u_idx.size(); ++i)
        {
            fsFinal << u_idx[i];
        }

        fsFinal << "]";

        fs.release();
        fsFinal.release();
    }
    else
    {

    }

}

int ShapeDescriptor::classify(const vector<Point> &src)
{
    if (trainFlag)
    {
        cerr << "\nFor use this func open class with flag for use. Fail!\n";
        return -100;
    }

    // contour sampling
    Mat Ssrc;
    ximgproc::contourSampling(src, Ssrc, 32);

    Ssrc -= mean(Ssrc); //minus centroid coord
    Ssrc.convertTo(Ssrc, CV_64FC2);

    //classify
    ximgproc::ContourFitting fit;
    fit.setFDSize(16);
    Mat t;
    vector<double> dist;
    for (int i = 0; i < avgDesc.size(); ++i)
    {
        double d;
        // compare matrix for the same
        MatExpr exp = (Ssrc != avgDesc[i]);
        if (countNonZero(exp) == 0)
        {
            dist.push_back(0);
            break;
        }

        fit.estimateTransformation(Ssrc, avgDesc[i], t, &d, false);
        std::cout << "\ndist to " << u_idx[i] << " = " << d << "\n";
        dist.push_back(d);
    }

    // min distance
    vector<double>::iterator min_it = min_element(dist.begin(), dist.end());
    int min_idx = distance(dist.begin(), min_it);
    double min_dist = (double)*min_it;

    if (min_dist > 0.35)
        return -1; // not found
    else
    {
        return u_idx[min_idx];
    }

}

int ShapeDescriptor::trainClassify(const vector<Point> &src)
{
    if (!trainFlag)
    {
        cerr << "\nFor use this func open class with flag for train. Fail!\n";
        return -100;
    }

    // contour sampling
    Mat Ssrc;
    ximgproc::contourSampling(src, Ssrc, 32);

    Ssrc -= mean(Ssrc); //minus centroid coord
    Ssrc.convertTo(Ssrc, CV_64FC2);
    //[debug print]
    //cout << "Ssrc = " << "\n";
    //std::cout << Ssrc << "\n";
    //[\debug print]

    //classify
    ximgproc::ContourFitting fit;
    fit.setFDSize(16);
    Mat t;
    vector<double> dist;
    for (int i = 0; i < avgDesc.size(); ++i)
    {
        double d;
        // compare matrix for the same
        MatExpr exp = (Ssrc != avgDesc[i]);
        if (countNonZero(exp) == 0)
        {
            dist.push_back(0);
            break;
        }

        fit.estimateTransformation(Ssrc, avgDesc[i], t, &d, false);
        std::cout << "dist to " << u_idx[i] << " = " << d << "\n";
        dist.push_back(d);
    }

    // min distance
    vector<double>::iterator min_it = min_element(dist.begin(), dist.end());
    int min_idx = distance(dist.begin(), min_it);
    double min_dist = (double)*min_it;

    if (min_dist > 0.35)
        return -1; // not found
    else
    {
        bool same{ false };
        for (int i = 0; i < mat.size(); ++i)
        {
            if (idx[i] == u_idx[min_idx])
            {
                MatExpr exp = (Ssrc != mat[i]);
                if (countNonZero(exp) == 0)
                {
                    same = true;
                    break;
                }
            }
        }
        // if not matrix the same, add new contour in DB
        if (same == false)
        {

            Mat Tsrc;
            ximgproc::transformFD(Ssrc, t, Tsrc, false);
            mat.push_back(Tsrc);
            idx.push_back(u_idx[min_idx]);
        }

        return u_idx[min_idx];
    }
}

void ShapeDescriptor::addShape(const vector<Point> &addContour, int label)
{
    if (!trainFlag)
    {
        cerr << "\nFor use this func open class with flag for train. Fail!\n";
        exit;
    }

    bool flag {false};

    fs.open(path, FileStorage::READ);

    if (fs.isOpened())
    {
        for (auto a : u_idx)
        {
            if (a == label)
            {
                std::cout << "\nCode " << std::to_string(label) << "alredy exist" << "\n\n";
                flag = true;
                fs.release();
                exit;
            }

        }

        fs.release();
    }
    else
    {
        cerr << "Failed to open " << path << "\n";
    }

    if (!flag)
    {

        // contour sampling
        Mat Sdst;
        ximgproc::contourSampling(addContour, Sdst, 32);

        Sdst -= mean(Sdst); //minus centroid coord

        //calculate Fourier descriptor
        /*Mat Fdst;
        dft(Sdst, Fdst, DFT_SCALE | DFT_REAL_OUTPUT);
        ximgproc::fourierDescriptor(Sdst, Fdst);*/
        //std::cout << "\n Fdst = " << "\n";
        //print(Fdst);

        Sdst.convertTo(Sdst, CV_64FC2);

        //add to mat
        mat.push_back(Sdst);
        idx.push_back(label);

        fs.release();

    }
}
