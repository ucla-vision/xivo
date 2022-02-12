#pragma once
/* ********************************* FILE ************************************/
/** \file    pose.h
 *
 * \brief    This header contains the Pose<T> class which represents 3D rigid transforms as a unit quaternion and translation
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - cuda enabled
 * - tested by test_pose.cpp
 *
 * Initialization using a rotation matrix is allowed but not ideal. Such conversions can give errors if the input isnt a rotation.
 *
 * \todo
 * - how to treat initialization by non rotation matrixes or nans? non rotation matrixes will be convertet to a rotation, but not in a good way. Nans give nans.
 * - convert from implicitly forming the rotation matrix to using quaternion algebra
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2014-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <utils/cvl/matrix.h>
#include <utils/cvl/rotation_helpers.h>
#include <vector>
namespace cvl{

/**
 * @brief The Pose<T> class
 * Quaternion represented rigid transform
 * and simplifies managing rotations by wrapping a quaternion
 *
 *
 * Initialization on bad rotation will assert
 * or give undefined behaviour( often the closest projection
 * onto the manifold of unit quaternions/rotations
 *
 *
 *  In the following, the Quaternions are laid out as 4-vectors, thus:

   q[0]  scalar part.
   q[1]  coefficient of i.
   q[2]  coefficient of j.
   q[3]  coefficient of k.


 q=(cos(alpha/2),sin(alpha/2)N)
 N= rotation axis
 *
 * This is the ceres& mathematical default
 *  but opengl libraries sometimes change the order of quaternion elements
 * There is no intrinsic direction of a transform Xa = Pab*Xb
 * Always specify a,b
 *
 *
 *
 *
 *
 * Tested by posetest.cpp
 */
template<class T>
class Pose {
public:



    __mlib_host_device
    /**
         * @brief Pose initializes as a identity transform
         */
    explicit Pose(){        q=Vector4<T>(1.0,0.0,0.0,0.0);        t=Vector3<T>(0.0,0.0,0.0);    }
    __mlib_host_device
    static Pose Identity(){return Pose();}

    /**
         * @brief Pose
         * @param Pose<U> converting constructor
         */
    template<class U>
    __mlib_host_device
    Pose(const Pose<U>& p){        q=Vector4<T>(p.q);        t=Vector3<T>(p.t);    }

    __mlib_host_device
    /**
         * @brief Pose
         * @param q_ unit quaternion
         * @param t_
         */
    Pose(const Vector4<T>& q_, const Vector3<T>& t_){        q=q_;        t=t_;    }

    __mlib_host_device
    /**
         * @brief Pose copies
         * @param q_ unit quaternion pointer
         * @param t_
         */
    explicit Pose(const T* q_, const T* t_, bool checked){
        for(int i=0;i<4;++i){q[i]=q_[i];} for(int i=0;i<3;++i){t[i]=t_[i];}
    }
    __mlib_host_device
    /**
         * @brief Pose copies
         * @param qt
         */
    explicit Pose(const T* qt, bool checked){
        for(int i=0;i<4;++i){q[i]=qt[i];} for(int i=0;i<3;++i){t[i]=qt[i+4];}
    }

    // user must verify that the matrix is a rotation separately
    __mlib_host_device
    /**
         * @brief Pose
         * @param R Rotation matrix
         * @param t_
         */
    Pose (const Matrix3x3<T>& R, const Vector3<T>& t_){
        q=getRotationQuaternion(R);        t=t_;
    }
    __mlib_host_device
    /**
         * @brief Pose translation 0
         * @param R rotation matrix
         */
    Pose (const Matrix3x3<T>& R){ q=getRotationQuaternion(R);        t=Vector3<T>(0,0,0);}
    // Rotation is i<T>entity
    __mlib_host_device
    /**
         * @brief Pose identity rotation assumed
         * @param t_ translation vector
         */
    Pose (const Vector3<T>& t_){ q=Vector4<T>(T(1),T(0),T(0),T(0));        t=t_;         }

    __mlib_host_device
    /**
         * @brief Pose
         * @param P a 3x4 matrix with a top left part beeing a rotation matrix
         */
    Pose (const Matrix3x4<T>& P){ q=getRotationQuaternion(P.getRotationPart());        t=P.getTranslationPart();}
    __mlib_host_device
    /**
         * @brief Pose
         * @param P a 4x4 matrix with a top left part beeing a rotation matrix
         */
    Pose (const Matrix4x4<T>& P){ q=getRotationQuaternion(P.getRotationPart());        t=P.getTranslationPart();}
    __mlib_host_device
    ~Pose (){}
    __mlib_host_device
    static Pose<T> eye(){
        return Pose(Vector<T,4>(1.0,0.0,0.0,0.0),Vector<T,3>(0.0,0.0,0.0));
    }

    /**
         * @brief getRRef
         * @return a pointer to the first element of the quaternion in the pose
         */
    __mlib_host_device
    /**
         * @brief getRRef pointer to the quaternion
         * @return
         */
    T* getRRef() {return &q[0];}
    /**
         * @brief getTRefJ
         * @return a pointer to the first element of the translation in the pose
         */
    __mlib_host_device
    /**
         * @brief getTRef pointer to the translation
         * @return
         */
    T* getTRef(){return &t[0];}
    __mlib_host_device
    /**
         * @brief setT set the translation vector, note this is in the camera coordinate system
         * @param t_
         */
    void setT(const Vector3<T>& t_){t=t_;}
    __mlib_host_device
    /**
         * @brief setQuaternion set the quaternion, note t is in the new coordinate system
         * @param q_
         */
    void setQuaternion(const Vector4<T>& q_){q=q_;}




    __mlib_host_device
    /**
         * @brief operator * applies the transform on the point
         * @param ins
         * @return
         *
         * based on the quaternion rotation of ceres, well why though? they basically just create a rotation matrix && apply it...
         */
    Vector3<T> operator*(const Vector3<T>& ins) const{
        // faster to get a 4x4 mat then multiply it onto the ext vector
        Matrix4x4<T> M=get4x4();
        Vector4<T> oh=M*Vector4<T>(ins[0],ins[1],ins[2],1);
        return Vector3<T>(oh[0],oh[1],oh[2]);
    }

    __mlib_host_device
    /**
         * @brief operator * apply the pose from the left!
         * @param rhs
         * @return
         */
    Pose<T> operator*(const Pose<T>& rhs) const{
        return Pose(get4x4()*rhs.get4x4());
        //return Pose(QuaternionProduct(q,rhs.q),QuaternionRotate(q,rhs.t));
    }
    __mlib_host_device
    /**
         * @brief inverse, note uses that the rotation inverse is its transpose
         * @return
         */
    Pose<T> inverse() const{
        //assert(isnormal());
        //Matrix3x3<T> Ri=getR().transpose();
        Vector<T,4> qi=q;qi[0]=-qi[0];
        Matrix3x3<T> Ri=getRotationMatrix(qi); // replace later for elegance...
        Vector3<T> ti=-Ri*getT();

        return Pose(qi,ti);
    }
    __mlib_host_device
    /**
         * @brief invert, note uses that the rotation inverse is its transpose
         * @return
         */
    void invert() {
        //assert(isnormal());
        //Matrix3x3<T> Ri=getR().transpose();
        Vector<T,4> qi=q;qi[0]=-qi[0];
        Matrix3x3<T> Ri=getRotationMatrix(qi); // replace later for elegance...
        Vector3<T> ti=-Ri*getT();
        q=qi;t=ti;
    }
    __mlib_host_device
    /**
         * @brief getR
         * @return the rotation matrix
         */
    Matrix3x3<T> getR() const{
        return getRotationMatrix(q);
    }
    __mlib_host_device
    /**
         * @brief rotation
         * @return the rotation matrix
         */
    Matrix3x3<T> rotation() const{return getR();}
    __mlib_host_device
    /**
         * @brief noRotation is the rotation matrix identity
         * @return
         */
    bool noRotation() const{
        if(q[0]!=1) return false;
        for(int i=1;i<4;++i)
            if(q[i]!=0) return false;
        return true;
    }
    __mlib_host_device
    /**
         * @brief isIdentity
         * @return is the pose a identity transform
         */
    bool isIdentity() const{
        if(!noRotation()) return false;
        for(int i=0;i<3;++i)
            if(t[i]!=0) return false;
        return true;
    }
    __mlib_host_device
    /**
         * @brief getT
         * @return the translation
         */
    Vector3<T> getT() const{return t;}
    __mlib_host_device
    /**
         * @brief translation
         * @return the translation
         */
    Vector3<T> translation() const{return t;}
    __mlib_host_device
    /**
         * @brief scaleT applies a scaling to the translation
         * @param scale
         */
    void scaleT(T scale){t*=scale;}
    __mlib_host_device
    /**
         * @brief get3x4
         * @return the 3x4 projection matrix corresp to the rigid transform
         */
    Matrix3x4<T> get3x4() const{return cvl::get3x4(getR(),getT());}
    __mlib_host_device
    /**
         * @brief get4x4
         * @return the 4x4 maxtrix rep of the rigid transform
         */
    Matrix4x4<T> get4x4() const{return cvl::get4x4(getR(),getT());}


    /**
         * @brief getAngle
         * @return the angle of the rotation in radians
         */
    T getAngle() const{
        if(std::abs(q[0]-1)<1e-6) return 0; // std::acos is numerically crap near the 1
        return 2.0*std::acos(q[0]);
    }

    /// get the position of the camera center in world coordinates
    __mlib_host_device
    /**
         * @brief getTinW
         * @return the camera center in world coordinates
         */
    Vector3<T> getTinW() const{
        Matrix3x3<T> R=getR();
        Vector3<T> t=getT();
        return -R.transpose()*t;
    }
    /**
         * @brief getEulerAngles
         * @return the euler angles according to spec in getEulerAngles()
         */
    Vector3<T> getEulerAngles() const{
        Matrix3x3D R=getR();
        return cvl::getEulerAngles(R);
    }
    __mlib_host_device
    /**
         * @brief getEssentialMatrix
         * @return the normalized essential matrix. This function defines the definition of E used.
         */
    Matrix3x3<T> getEssentialMatrix() const{
        Vector3<T> t=getT();t.normalize();
        Matrix3x3<T> E=t.crossMatrix()*getR();
        assert(E.isnormal());
        T max=E.absMax();
        E/=max;
        return E;
    }
    __mlib_host_device
    /**
         * @brief distance
         * @param p
         * @return distance between two coordinate system centers
         */
    T distance(const Pose<T>& p) const{
        Pose<T> Pab=(*this)*p.inverse();return Pab.t.length();
    }
    /**
         * @brief angleDistance
         * @param p
         * @return the positive angle between two coordinate systems
         */
    T angleDistance(const Pose<T>& p) const{
        Pose<T> Pab=(*this)*p.inverse();
        double angle=Pab.getAngle();
        if(angle<0)return -angle;
        return angle;
    }
    /// returns true if no value is strange
    __mlib_host_device
    /**
         * @brief isnormal
         * @return true if the pose contains no nans or infs and the quaternion is a unit quaternion
         */
    bool isnormal() const{
        if (!q.isnormal()) return false;
        if (!t.isnormal()) return false;
        if(q.length()-1.0>1e-5) return false;
        return true;
    }


    __mlib_host_device
    /**
         * @brief getQuaternion
         * @return the quaternion
         */
    Vector4<T> getQuaternion() const{return q;}
    __mlib_host_device
    /**
         * @brief normalize ensures that the quaternion length is 1, helpful to counter numerical errors
         */
    void normalize(){
        q.normalize();
    }


    __mlib_host_device
    /**
         * @brief rotateInplace rotates but does not translate the point
         * @param x
         */
    void rotateInplace(Vector3<T>& x) const{
        // improve performance if relevant through quaternion rotation
        auto R=getR();
        x=R*x;
    }
    __mlib_host_device
    /**
         * @brief rotate
         * @param x
         * @return  the rotated but not translated vector
         */
    Vector3<T> rotate(const Vector3<T>& x) const{
        Vector3<T> xr=x;
        rotateInplace(xr);
        return xr;
    }

    //private:
    /// the unit quaternion representing the rotation, s,i,j,k
    Vector4<T> q;
    /// the translation: x' = (R(q)x) +t
    Vector3<T> t;
    //T filler;
};




/**
     * @brief apply a fast version of outs =P*ins
     * @param pose
     * @param ins
     * @param outs
     */
template<class T>
void apply(const Pose<T>& pose,
           const std::vector<Vector3D>& ins,
           std::vector<Vector3D>& outs){
    outs.resize(ins.size());
    Vector4<T> xh;
    Matrix4x4<T> M=pose.get4x4(); // much faster...
    for(unsigned int i=0;i<ins.size();++i){
        xh=ins[i].homogeneous();
        xh=M*xh;
        outs[i]=Vector3<T>(xh[0],xh[1],xh[2]);
    }
}



template<class T> Matrix<T,4,4> invertPoseMatrix(const Matrix<T,4,4>& pose){
    Matrix<T,3,3> R=pose.getRotationPart();
    Vector3<T> t=pose.getTranslationPart();
    R=R.transpose();
    t=-R*t;
    return get4x4(R,t);
}





template<class T> Pose<T> lookAt(const Vector3<T>& point, const Vector3<T>& from, const Vector3<T>& up0){
    // opt axis is point
    Vector3<T> up=up0;            up.normalize();
    Vector3<T> z=point -from;     z.normalize();
    Vector3<T> s=-z.cross(up);    s.normalize();
    Vector3<T> u=z.cross(s);     u.normalize();
    /*
        cout<<"s: "<<s<<endl;
        cout<<"u:  "<<u<<" uz"<<cvl::dot(u,z)<<endl;
        cout<<"up: "<<up<<endl;
        cout<<"z: "<<z<<endl;
*/
    // u=cross f,s

    Matrix3x3D R(s[0],s[1],s[2],
            u[0],u[1],u[2],
            z[0],z[1],z[2]);
    //cout<<"det: "<<R.determinant()<<endl;

    return Pose<T>(R,-R*from);
    /*
        void lookattest(){
            cv::namedWindow("test");
            std::shared_ptr<PointCloudViewer> shower=PointCloudViewer::start();




            std::vector<PoseD> poses;
            poses.push_back(PoseD());

            std::vector<Vector3> xs;std::vector<Color> cs;
            xs.push_back(Vector3(0,0,0));
            xs.push_back(Vector3(1,0,0));
            xs.push_back(Vector3(0,1,0));
            xs.push_back(Vector3(0,0,1));
            Vector3D from(1,1,0);
            Vector3D point(0,0,1);
            Vector3D up(0,1,0);
            xs.push_back(from);
            xs.push_back(point);
            cs.push_back(Color::cyan());
            cs.push_back(Color::red());
            cs.push_back(Color::green());
            cs.push_back(Color::blue());
            cs.push_back(Color::yellow());
            cs.push_back(Color::black());

            PoseD la=lookAt2(point,from,up);

            poses.push_back(la);
            Matrix3x3D R=la.getR();
            cout<<"R: "<<R<<endl;
            Vector4 q=getRotationQuaternion(R);
            cout<<"q: "<<q<<endl;
            cout<<"T: "<<la.getT()<<endl;
            cout<<"Tw: "<<la.getTinW()<<endl;

            shower->setPointCloud(xs,cs,poses);
            cv::waitKey(0);
        }
        */

}


/// convenience alias for the standard pose
typedef Pose<double> PoseD;



}// en<T> namespace cvl


