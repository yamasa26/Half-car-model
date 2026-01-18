#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;

typedef Matrix<double, 8, 1> Vector8d;

struct Params {
    double ms, Is, mu1, mu2;    // 質量[kg], 慣性モーメント[kg*m^2], バネ下質量[kg]
    double ks1, ks2, kt1, kt2; // バネ定数, タイヤ剛性[N/m]
    double cs1, cs2;           // 減衰係数[N*s/m]
    double l1, l2, h;          // 重心から車軸の距離[m], 重心高[m]
};

class HalfCarModel {
public:
    Matrix4d M, C, K;

    void updateMatrices(const Params& p) {
        M.setZero();
        M.diagonal() << p.ms, p.Is, p.mu1, p.mu2;

        K << (p.ks1 + p.ks2),             (-p.ks1 * p.l1 + p.ks2 * p.l2), -p.ks1,          -p.ks2,
             (-p.ks1 * p.l1 + p.ks2 * p.l2), (p.ks1 * p.l1 * p.l1 + p.ks2 * p.l2 * p.l2), p.ks1 * p.l1, -p.ks2 * p.l2,
             -p.ks1,                       p.ks1 * p.l1,                  (p.ks1 + p.kt1), 0,
             -p.ks2,                      -p.ks2 * p.l2,                   0,              (p.ks2 + p.kt2);

        C << (p.cs1 + p.cs2),             (-p.cs1 * p.l1 + p.cs2 * p.l2), -p.cs1,          -p.cs2,
             (-p.cs1 * p.l1 + p.cs2 * p.l2), (p.cs1 * p.l1 * p.l1 + p.cs2 * p.l2 * p.l2), p.cs1 * p.l1, -p.cs2 * p.l2,
             -p.cs1,                       p.cs1 * p.l1,                   p.cs1,          0,
             -p.cs2,                      -p.cs2 * p.l2,                   0,              p.cs2;
    }

    Vector4d calculateExternalForce(const Params& p, double accel) {
        //pitching moment: M = m * a * h (反時計回りが正)
        return Vector4d(0.0, p.ms * accel * p.h, 0.0, 0.0);
    }
};

class Simulator {
public:
    HalfCarModel model;
    Params p;

    Vector8d f(const Vector8d& x, double t, double accel) {
        Vector4d q = x.segment<4>(0);
        Vector4d dq = x.segment<4>(4);
        Vector4d F = model.calculateExternalForce(p, accel);
        Vector4d ddq = model.M.diagonal().asDiagonal().inverse() * (F - model.C * dq - model.K * q);

        Vector8d dxdt;
        dxdt << dq, ddq;
        return dxdt;
    }

    Vector8d rk4(const Vector8d& x, double t, double dt, double accel) {
        Vector8d k1 = f(x, t, accel);
        Vector8d k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt, accel);
        Vector8d k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt, accel);
        Vector8d k4 = f(x + dt * k3, t + dt, accel);
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
};

int main() {
    Simulator sim;
    sim.p = {1200.0, 2000.0, 40.0, 40.0, 25000.0, 25000.0, 150000.0, 150000.0, 1500.0, 1500.0, 1.2, 1.3, 0.5};
    sim.model.updateMatrices(sim.p);

    std::ofstream file("simulation_results.csv");
    file << "time,ys,theta,yu1,yu2,v_abs,x_abs\n";

    Vector8d x = Vector8d::Zero();
    double t = 0.0, dt = 0.001, v_abs = 0.0, x_abs = 0.0;

    for (int i = 0; i < 5000; ++i) {
        double accel = (t > 0.5 && t < 2.5) ? 3.0 : (t > 3.0 && t < 4.0) ? -6.0 : 0.0;
        file << t << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "," << v_abs << "," << x_abs << "\n";
        
        x = sim.rk4(x, t, dt, accel);
        v_abs += accel * dt;
        x_abs += v_abs * dt;
        t += dt;
    }
    return 0;
}