# VQLS CFD Pressure Solve Results


---

## Systems Tested

> The backend used was the **Classiq simulator (`simulator_statevector`)**.

$$
b = H_0 H_1 H_2 \ket{0}
$$

### **System 1 – ps1_test**

$$
A_1 = 0.55I_0 + 0.225(I_0 Z_1 I_2) + 0.225(I_0 I_1 Z_2)
$$

### **System 2 – ps2_test**
$$
A_2 = 0.6I_0 + 0.2(Z_0 I_1 I_2) + 0.2(I_0 Z_1 Z_2)
$$

### **System 3 – ps3_test**
$$
A_3 = 0.6I_0 + 0.2(Z_0 I_1 I_2) + 0.2(I_0 Y_1 Z_2)
$$

---

## Results Overview


| Test | Iterations | Overlap | MSE | Cosine Similarity |
|------|-------------|----------|--------|--------------------|
|A1 | 107 | 0.8537 | 0.04196 | 0.7666 |
|A2 | 103 | 0.8597 | 0.02977 | 0.7456 |
|A3 | 110 | 0.8648 | -0.01879 | 0.9814 |




![](./images/ps1_test.png)

![](./images/ps2_test.png)

![](./images/ps3_test.png)
