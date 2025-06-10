# BEACON
Project in Seoul National University's Intelligent Computing System Design Project AI for Social Good (2025-1)

# How to Start
This project uses [BML](https://github.com/KevinHoarau/BML) and [BGNN](https://github.com/KevinHoarau/BGNN)

Before executing experiments and demo file, followings are required.

```bash
BML library installation through BML github page

// after installing BML
>>> python3 collect_transform.py   // Collects BGP data using BML
>>> python3 transform_to_pyg.py    // Extracts BGP graphs from BGP data
```

# Experiments
This project provides 3 experiments.
1. Accuracy of BEACON
```bash
>>> python3 eval.py
```
2. Train Efficiency of BEACON
```bash
>>> python3 measure_train_time.py
``` 
3. Test Efficiency of BEACON
```bash
>>> python3 measure_test_time.py
```

# Demo
Makes sure that data_pyg.pickle is generated through [How to Start](#How-to-Start)
```bash
>>> cd demo
>>> python3 generate_demo_model.py  // Can be skipped since output of this function is provided (`model_demo.pt`)
>>> python3 demo_infer.py
```
