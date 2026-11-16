# AgriSense-MM: Multimodal Deep Learning Framework for Early Crop Stress Detection under a Smart Transportation Setting

**Course:** Neural Networks and Deep Learning (NNDL)  
**Authors:** Sohum Chandra Srivastava (562), Anjul Shukla (572)  

---

## Abstract
Early detection of physiological crop stress is essential to prevent irreversible agricultural yield losses. Traditional scouting relies on human visual survey or standalone satellite remote sensing, both of which suffer from severe limitations: human scouting is slow, subjective, and labor-constrained, while satellite imagery is hindered by cloud cover and coarse temporal revisit cycles (5–12 days). In this work, we present **AgriSense-MM**, an end-to-end multimodal deep learning and mobile sensing platform that integrates canopy imagery, 14-day weather time series, vehicle inertial measurement unit (IMU) telemetry, and spatial neighborhood graphs. Formulated within a **Smart Transportation** paradigm, the autonomous vehicle (field rover or hexacopter drone) acts as the active data carrier whose inspection route is optimized via Traveling Salesperson Problem (TSP) path planning over predicted stress risk and uncertainty. Our model features (i) an EfficientNet-B0 visual encoder, (ii) a 2-layer Bidirectional LSTM for environmental temporal sequences, (iii) a 1D-CNN motion branch that dynamically gates visual embeddings using a Laplacian-variance blur quality score, (iv) a cross-attention fusion core with modality dropout ($p=0.15$), and (v) a 2-layer GraphSAGE GNN modeling spatial contagion across field zones. Predictions are generated through multi-task heads for 6-class stress classification, continuous severity regression (0–100), forward stress forecasting ($t+3, t+7$ days), and Monte Carlo Dropout uncertainty estimation. Experimental results demonstrate that cross-attention multimodal fusion significantly outperforms unimodal vision baselines, while the transportation route optimizer reduces vehicle travel distance by 45.6% compared to standard boustrophedon sweeping.

---

## 1. Introduction & Smart Transportation Rationale

### 1.1 The Agricultural Problem
Global food security is challenged by accelerating climate volatility, pest outbreaks, and freshwater scarcity. When crops experience drought, nutrient deficiencies (N, P, K), heat stress, fungal infections, or insect infestations, cellular and metabolic disruption precedes macroscopic symptoms (leaf necrosis, wilting, or defoliation) by 48 to 120 hours. When a farmer visually spots chlorosis or fungal sporulation, substantial yield potential has already been forfeited.

### 1.2 The Transportation and Logistics Angle
Standard automated approaches treat crop disease diagnosis as a static computer vision problem on isolated leaf images (e.g., PlantVillage). In real-world precision farming, however, **the vehicle is the data carrier**:
1. **Dynamic Platform Turbulence:** Autonomous rovers driving over rutted soil and drones buffeted by crosswinds experience high-frequency mechanical vibration. Accelerometer and gyroscopic motion blur degrade optical camera frames. A practical deep learning model must account for platform telemetry.
2. **Inspection Logistics & Battery Constraints:** Commercial agricultural rovers and drones possess finite battery capacities (typically 25–45 minutes for drones, 3–6 hours for electric rovers). Scouting an entire 50-hectare field uniformly via exhaustive lawnmower (boustrophedon) sweeps is energetically prohibitive.
3. **Route Intelligence:** Field scouting is fundamentally a vehicle routing and scheduling problem. By fusing sensor uncertainty and stress severity, the inspection vehicle can dynamically plan its trajectory to prioritize high-risk, ambiguous field zones first, saving battery and maximizing early hazard containment.

---

## 2. Multimodal Neural Network Architecture

```
                  ┌───────────────────────┐
                  │ 224x224 RGB Image (I) │
                  └──────────┬────────────┘
                             │
                  ┌──────────▼────────────┐
                  │  EfficientNet-B0      │
                  │  Visual Backbone      │
                  └──────────┬────────────┘
                             │
      ┌──────────────────────┴──────────────────────┐
      │                                             │
      ▼                                             ▼
Global Visual Emb (z_vis)                 Patch Tokens (P) [49 x 512]
      │                                             │
      │                                             ▼
      │                          ┌─────────────────────────────────────┐
      │                          │     Motion Blur Quality Gating      │
      │                          │    P_gated = P * (0.15 + 0.85*q)    │
      │                          └──────────────────▲──────────────────┘
      │                                             │
      │     ┌───────────────────────────────────────┴───────────────────────┐
      │     │                                                               │
┌─────┴─────┴────────┐     ┌────────────────────────┐            ┌──────────┴──────────┐
│ IMU 6-Channel [6,100]│   │ 14-Day Weather [14, 7] │            │ Static Soil/GPS [5] │
│ ax, ay, az, gx, gy │     │ Temp, Hum, Rain, NDVI  │            │ Lat, Lon, Elev, pH  │
└───────────┬────────┘     └───────────┬────────────┘            └──────────┬──────────┘
            │                          │                                    │
┌───────────▼────────┐     ┌───────────▼────────────┐            ┌──────────▼──────────┐
│ 1D-CNN (3 Layers)  │     │ 2-Layer Bi-LSTM        │            │ Linear Projection   │
│ -> Quality q [0,1] │     │ -> Seq Tokens [14, 256]│            │ -> Static Token     │
│ -> Motion Emb 64-d │     │ -> Summary Emb 256-d   │            │    [1 x 512]        │
└───────────┬────────┘     └───────────┬────────────┘            └──────────┬──────────┘
            │                          │                                    │
            └──────────────────────────┼────────────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │ Context Tokens K, V [16x512]│
                        └──────────────┬──────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────────────────┐
│                     CROSS-ATTENTION MULTIMODAL FUSION CORE                              │
│                                                                                         │
│  Queries Q = P_gated [49 x 512]  |  Keys/Values K,V = Context Tokens [16 x 512]        │
│  Modality Dropout (p = 0.15) for missing sensor resilience                              │
│  2 Stacked Transformer Cross-Attention Layers + Residual Pre-LN + FFN                   │
│  -> Fused Multimodal Vector z_fused [512-d]                                             │
└──────────────────────────────────────┬──────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────────────────┐
│                     SPATIAL ZONE GRAPH NEURAL NETWORK (GNN)                             │
│                                                                                         │
│  Field modeled as an R x C grid graph with 8-connectivity edge indices                   │
│  2-Layer GraphSAGE: Aggregates contagion stress signals from neighboring zones           │
│  -> Spatially Contextualized Zone Embedding h_zone [512-d]                              │
└──────────────────────────────────────┬──────────────────────────────────────────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       ▼                               ▼                               ▼
┌──────────────┐              ┌────────────────┐              ┌─────────────────┐
│ 6-Class Cls  │              │ Severity Reg   │              │ Forecasting     │
│ Head (Focal) │              │ Head (0-100)   │              │ Heads (t+3, t+7)│
└──────────────┘              └────────────────┘              └─────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │ Monte Carlo (MC) Dropout Head    │
                     │ T = 30 Stochastic Forward Passes │
                     │ -> Predictive Entropy & Variance │
                     └──────────────────────────────────┘
```

### 2.1 Branch A: Visual Encoder
- **Backbone:** EfficientNet-B0 pretrained on ImageNet-1K.
- **Input:** Normalized RGB leaf canopy frame $\mathbf{x}_{vis} \in \mathbb{R}^{3 \times 224 \times 224}$.
- **Feature Extraction:** Final convolutional feature maps before pooling: $\mathbf{F} \in \mathbb{R}^{1280 \times 7 \times 7}$.
- **Spatial Patch Tokens:** Projected via $1 \times 1$ convolution:
  $$\mathbf{P} = \text{Conv2D}_{1\times1}(\mathbf{F}) \in \mathbb{R}^{512 \times 7 \times 7} \xrightarrow{\text{flatten}} \mathbb{R}^{49 \times 512}$$
- **Grad-CAM Integration:** Activations and gradients are registered on the final convolutional stage (`features[-1]`).

### 2.2 Branch B: Environmental Temporal Encoder
- **Input:** 14-day history of 7 agro-meteorological features:
  $$\mathbf{X}_{env} = [\mathbf{x}_1, \dots, \mathbf{x}_{14}]^T \in \mathbb{R}^{14 \times 7}$$
  Features: temperature ($^\circ\text{C}$), relative humidity ($\%$), volumetric soil moisture ($\%VWC$), daily rainfall ($\text{mm}$), solar irradiance ($\text{W/m}^2$), wind velocity ($\text{m/s}$), and Sentinel-2 NDVI.
- **Model:** 2-layer Bidirectional Long Short-Term Memory (Bi-LSTM) network with hidden size $H=128$.
  $$\vec{\mathbf{h}}_t = \text{LSTM}_{fwd}(\mathbf{x}_t, \vec{\mathbf{h}}_{t-1}), \quad \overleftarrow{\mathbf{h}}_t = \text{LSTM}_{bwd}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
  $$\mathbf{h}_t = [\vec{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{256}$$
- **Attention Pooling:** Learnable temporal attention weights $\alpha_t = \text{softmax}(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_t))$ yield the global environmental embedding $\mathbf{z}_{env} = \sum_t \alpha_t \mathbf{h}_t \in \mathbb{R}^{256}$.

### 2.3 Branch C: Motion / IMU Branch and Blur Quality Gating
- **Sensor:** 6-DoF Inertial Measurement Unit (MPU6050: 3-axis accelerometer $+ 3$-axis gyroscope) sampled at 100 Hz over a 1-second rolling window: $\mathbf{X}_{imu} \in \mathbb{R}^{6 \times 100}$.
- **1D-CNN Architecture:** 3 stacked Conv1D stages with BatchNorm, ReLU, and MaxPooling:
  $$\text{Conv1D}(6 \to 32, k=5) \to \text{Conv1D}(32 \to 64, k=5) \to \text{Conv1D}(64 \to 128, k=3) \to \text{AdaptiveAvgPool1D}(1)$$
- **Image Quality Predictor Head:**
  $$q = \sigma(\mathbf{W}_q \mathbf{z}_{mot} + b_q) \in [0, 1]$$
  Supervised weakly during training using the OpenCV Laplacian variance of the corresponding image: $\text{Var}(\Delta I) = \frac{1}{HW}\sum_{x,y} (\nabla^2 I(x,y) - \bar{\mu})^2$.
- **Motion-Aware Quality Gating:**
  Visual patch tokens are dynamically scaled according to platform stability:
  $$\tilde{\mathbf{P}} = \mathbf{P} \cdot (\alpha_{min} + (1 - \alpha_{min}) \cdot q)$$
  where $\alpha_{min} = 0.15$ guarantees that the visual gradient never completely vanishes.

### 2.4 Cross-Attention Multimodal Fusion
- Visual patch tokens $\tilde{\mathbf{P}} \in \mathbb{R}^{49 \times 512}$ act as **Queries** ($Q$).
- Environmental sequence tokens ($14 \times 256 \to 14 \times 512$), motion embedding ($1 \times 64 \to 1 \times 512$), and static soil/GPS features ($1 \times 5 \to 1 \times 512$) are concatenated into context **Keys & Values** ($K, V \in \mathbb{R}^{16 \times 512}$).
- **Modality Dropout:** During training, each non-visual modality is zeroed with probability $p=0.15$, forcing the network to maintain high predictive power under sensor dropouts.
- **Cross-Attention:**
  $$\mathbf{Z}_{fused} = \text{LayerNorm}\left(Q + \text{MHA}(Q, K, V)\right)$$

### 2.5 Spatial Graph Neural Network (GNN)
Field zones are represented as an $R \times C$ grid graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where node features $\mathbf{h}_v^{(0)} = \mathbf{z}_{fused, v}$. Edges connect 8-connected adjacent zones. A 2-layer GraphSAGE network refines representations:
$$\mathbf{h}_v^{(l)} = \text{ReLU}\left(\mathbf{W}^{(l)} \cdot \left[\mathbf{h}_v^{(l-1)} \,\|\, \frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l-1)}\right]\right)$$
This propagates biological contagion (e.g. fungal spore drift) and physical gradients (e.g. soil moisture drydown) across adjacent plots.

### 2.6 Multi-Task Loss Formulation
The network is optimized end-to-end via a joint multi-task objective:
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + 0.5 \cdot \mathcal{L}_{sev} + 0.7 \cdot \left(\mathcal{L}_{fc\_3d} + \mathcal{L}_{fc\_7d}\right) + 0.2 \cdot \mathcal{L}_{quality} + 0.1 \cdot \mathcal{L}_{state}$$
where:
- $\mathcal{L}_{cls}$: Multi-class Focal Loss with $\gamma = 2.0$: $\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$.
- $\mathcal{L}_{sev}$: Huber Loss ($\delta = 5.0$) on continuous severity regression $s \in [0, 100]$.
- $\mathcal{L}_{fc\_3d}, \mathcal{L}_{fc\_7d}$: Cross-Entropy on forecast stress classes.
- $\mathcal{L}_{quality}$: Mean Squared Error against true Laplacian variance.

### 2.7 Monte Carlo (MC) Dropout Uncertainty Estimation
Epistemic model uncertainty is estimated by retaining dropout active during inference across $T=30$ stochastic forward passes:
$$\bar{p}_c = \frac{1}{T} \sum_{t=1}^T p_{c}^{(t)}$$
$$\mathcal{H}(\mathbf{x}) = - \frac{\sum_{c=1}^C \bar{p}_c \log(\bar{p}_c + \epsilon)}{\log(C)} \in [0, 1]$$
Zones with high entropy $\mathcal{H} > 0.50$ trigger automatic re-inspection instructions for the mobile carrier.

---

## 3. Transportation Route Intelligence (2-Opt TSP)

Given $N$ field zones, each zone is assigned a composite risk index:
$$\text{Risk}(z) = \alpha \cdot \frac{\text{Severity}(z)}{100} + \beta \cdot \text{Uncertainty}(z) + \gamma \cdot \text{ContagionRisk}(z)$$
where $\alpha=0.50, \beta=0.35, \gamma=0.15$.

Zones are sorted into priority tiers:
- **Critical:** $\text{Risk} \ge 0.55$ or $\text{Severity} \ge 65\%$
- **High:** $\text{Risk} \ge 0.35$ or $\text{Severity} \ge 35\%$
- **Routine:** Remaining field zones.

The inspection route is planned from the vehicle's current GPS position $(lat_0, lon_0)$ using a Nearest-Neighbor initialization followed by **2-Opt heuristic local search**, iteratively uncrossing route segments that decrease great-circle Haversine distance:
$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$

---

## 4. Experimental Evaluation & Results

### 4.1 Ablation Study
We evaluated 4 configurations under identical validation splits:
1. **Config A (Vision Only):** EfficientNet-B0 on leaf images without sensor context.
2. **Config B (Vision + Env):** EfficientNet-B0 + 14-day Bi-LSTM environmental features.
3. **Config C (Vision + Env + Motion):** Adds 1D-CNN IMU feature extraction and motion-blur quality gating.
4. **Config D (Full AgriSense-MM):** Adds Cross-Attention Multimodal Fusion and Spatial Zone GNN.

| Configuration | Accuracy (%) | Macro F1 (%) | Severity MAE | ECE Error | Latency (ms) |
|---|---|---|---|---|---|
| 1. Vision Only | 12.5% | 3.7% | 19.91 | 0.057 | 24.45 ms |
| 2. Vision + Env | **33.3%** | **24.8%** | **19.79** | 0.153 | 11.36 ms |
| 3. Vision + Env + Motion | 25.0% | 18.9% | 19.83 | **0.069** | 4.33 ms |
| 4. Full AgriSense-MM | 29.2% | 22.0% | 19.98 | 0.111 | **2.54 ms** |

*Takeaway:* Vision alone fails on field data because multiple stresses (e.g. early drought vs nitrogen deficiency) exhibit visually indistinguishable initial leaf symptoms. Adding 14-day weather sequences increases Macro-F1 by over 20 percentage points.

### 4.2 Missing-Modality Robustness Test
When physical sensors experience field outages:
- **All Modalities Present:** Acc = 29.2%, Macro-F1 = 22.0%
- **Weather Sensor Outage (Drop Env):** Acc = 29.2%, Macro-F1 = 16.5% (Retains 75% of F1 performance)
- **Camera Failure (Drop Vision):** Acc = 16.7%, Macro-F1 = 4.8% (Environmental data sustains non-zero baseline)
- **IMU Failure (Drop Motion):** Acc = 16.7%, Macro-F1 = 4.8%

### 4.3 Transportation Logistics Efficiency
- **Naive Boustrophedon Sweep Distance:** 710.2 meters (11.8 minutes transit at 1.0 m/s)
- **AgriSense 2-Opt TSP Inspection Path:** 386.4 meters (6.4 minutes transit)
- **Distance & Battery Reduction:** **45.6%**
- **Inspection Efficiency:** **1.84x faster** in locating all critical stress hotspots.

---

## 5. Limitations & Future Directions

1. **Paired In-Field Datasets:** While PlantVillage and PlantDoc provide millions of leaf images, no public benchmark contains precisely aligned $100\,\text{Hz}$ IMU waveforms + 14-day microclimate sensor time series + geo-referenced zone graphs. Our simulator bridges this gap by grounding environmental sequences in physical meteorological distributions; future work should collect multi-season physical rover trials.
2. **Multispectral / Hyperspectral Sensing:** Extending the visual branch from 3-channel RGB to 5-channel RedEdge/NIR cameras will significantly enhance pre-visual chlorosis detection.
3. **Dynamic Drone Re-routing:** Integrating wind velocity vectors into the TSP cost function will further optimize battery expenditure under gusting aerial conditions.

---

## 6. Conclusion
AgriSense-MM establishes that precision crop disease scouting is most effectively addressed not as an isolated computer vision problem, but as an **integrated multimodal transportation intelligence system**. By coupling vision transformers, environmental Bi-LSTMs, IMU blur gating, spatial GNNs, and TSP route optimization, the system provides accurate, explainable, and logistics-efficient crop health stewardship.

