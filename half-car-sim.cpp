#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;

typedef Matrix<double, 8, 1> Vector8d;

struct Params
{
    std::string name;
    double ms, Is, mu1, mu2;   // 質量[kg], 慣性モーメント[kg*m^2], バネ下質量[kg]
    double ks1, ks2, kt1, kt2; // バネ定数, タイヤ剛性[N/m]
    double cs1, cs2;           // 減衰係数[N*s/m]
    double l1, l2, h;          // 重心から車軸の距離[m], 重心高[m]
};

Params getGR86() { return {"GR86", 1150.0, 1400.0, 45.0, 45.0, 30000.0, 35000.0, 200000.0, 200000.0, 2500.0, 2800.0, 1.28, 1.29, 0.45}; }
Params getLexusLS() { return {"LexusLS", 2000.0, 3500.0, 65.0, 65.0, 20000.0, 22000.0, 220000.0, 220000.0, 3500.0, 3800.0, 1.55, 1.57, 0.55}; }
Params getSamber() { return {"Samber", 650.0, 750.0, 35.0, 35.0, 15000.0, 25000.0, 160000.0, 160000.0, 1200.0, 1500.0, 0.95, 0.95, 0.70}; }

class HalfCarModel
{
public:
    Matrix4d M, C, K;

    void updateMatrices(const Params &p)
    {
        M.setZero();
        M.diagonal() << p.ms, p.Is, p.mu1, p.mu2;

        K << (p.ks1 + p.ks2), (-p.ks1 * p.l1 + p.ks2 * p.l2), -p.ks1, -p.ks2,
            (-p.ks1 * p.l1 + p.ks2 * p.l2), (p.ks1 * p.l1 * p.l1 + p.ks2 * p.l2 * p.l2), p.ks1 * p.l1, -p.ks2 * p.l2,
            -p.ks1, p.ks1 * p.l1, (p.ks1 + p.kt1), 0,
            -p.ks2, -p.ks2 * p.l2, 0, (p.ks2 + p.kt2);

        C << (p.cs1 + p.cs2), (-p.cs1 * p.l1 + p.cs2 * p.l2), -p.cs1, -p.cs2,
            (-p.cs1 * p.l1 + p.cs2 * p.l2), (p.cs1 * p.l1 * p.l1 + p.cs2 * p.l2 * p.l2), p.cs1 * p.l1, -p.cs2 * p.l2,
            -p.cs1, p.cs1 * p.l1, p.cs1, 0,
            -p.cs2, -p.cs2 * p.l2, 0, p.cs2;
    }

    
    Vector4d calculateExternalForce(const Params &p, double accel)
    {
        // ピッチングモーメント: M = m * a * h (反時計回りが正)
        return Vector4d(0.0, p.ms * accel * p.h, 0.0, 0.0);
    }
};

class Simulator
{
public:
    HalfCarModel model;
    Params p;

    // 2階微分方程式を1階の連立微分方程式に変換
    Vector8d f(const Vector8d &x, double t, double accel)
    {
        Vector4d q = x.segment<4>(0);
        Vector4d dq = x.segment<4>(4);
        Vector4d F = model.calculateExternalForce(p, accel);
        Vector4d ddq = model.M.diagonal().asDiagonal().inverse() * (F - model.C * dq - model.K * q);

        Vector8d dxdt;
        dxdt << dq, ddq;
        return dxdt;
    }

    // RK4の計算処理
    Vector8d rk4(const Vector8d &x, double t, double dt, double accel)
    {
        Vector8d k1 = f(x, t, accel);
        Vector8d k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt, accel);
        Vector8d k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt, accel);
        Vector8d k4 = f(x + dt * k3, t + dt, accel);
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    // csv出力
    void outputCSV(Params input_p)
    {
        this->p = input_p;
        model.updateMatrices(this->p);

        std::ofstream file("csv/" + this->p.name + "_sim.csv");
        file << "time,ys,theta,yu1,yu2,v_abs,x_abs\n";

        Vector8d x = Vector8d::Zero();
        double t = 0.0, dt = 0.001, v_abs = 0.0, x_abs = 0.0;

        const double target_speed = 65.0 / 3.6;
        bool is_braking = false;

        for (int i = 0; i < 9000; ++i)
        {
            double accel = 0.0;

            if (!is_braking)
            {
                if (v_abs < target_speed)
                {
                    accel = 3.3;
                }
                else
                {
                    is_braking = true;
                }
            }
            else
            {
                if (v_abs > 0.1)
                {
                    accel = -8.5;
                }
                else
                {
                    accel = 0.0;
                    v_abs = 0.0;
                }
            }

            file << t << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "," << v_abs << "," << x_abs << "\n";

            x = rk4(x, t, dt, accel);
            v_abs += accel * dt;
            x_abs += v_abs * dt;
            t += dt;
        }
        file.close();
    }
};

int main()
{
    Simulator sim;
    sim.outputCSV(getGR86());
    sim.outputCSV(getLexusLS());
    sim.outputCSV(getSamber());
    return 0;
}