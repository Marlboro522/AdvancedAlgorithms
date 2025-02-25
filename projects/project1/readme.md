# Transit Node Routing (TNR) Preprocessing Pippeline

## Overview
This repository implements a **preprocessing pipeline for Transit Node Routing (TNR)**, designed to efficiently prepare a road network for fast shortest-path computations. The preprocessing steps assign nodes to a grid, detect boundary-crossing roads, and identify transit nodes.

The pipeline is **generalizable** and can be used for **any road network**.

---

## Files and steps
| **Step** | **Description** | **Generated Files** |
|---------|---------------|---------------------|
| **1️⃣ Load Road Network** | Loads road data from OpenStreetMap | `colorado_springs.graphml` |
| **2️⃣ Create Grid** | Divides the area into a uniform square grid | `colorado_springs_grid.geojson` |
| **3️⃣ Assign Nodes to Grid** | Assigns each intersection (node) to a grid cell | `node_to_grid.json` |
| **4️⃣ Find Boundary Roads** | Identifies roads that cross grid cell boundaries | `boundary_crossing_edges.json` |
| **5️⃣ Identify Transit Node Candidates** | Marks nodes involved in boundary roads | `transit_nodes.json` |
| **6️⃣ Filter Final Transit Nodes** | Selects optimal transit nodes for fast routing | `final_transit_nodes.json` |

---

### **2Run the Preprocessing Pipeline**
To execute the preprocessing:
```bash
python preprocessing.py
```

---

## Usage
### **1️⃣ Load Road Network**
- The pipeline reads the **OSM road network** from a `.graphml` file.

### **2️⃣ Create Square Grid**
- A **30 × 30 square grid** is generated and saved as **GeoJSON**.

### **3️⃣ Assign Nodes to Grid**
- Each **road intersection (node)** is assigned to a grid cell.
- Stored in **`node_to_grid.json`**.

### **4️⃣ Detect Boundary-Crossing Roads**
- Any **road (edge) that crosses two grid cells** is saved in **`boundary_crossing_edges.json`**.

### **5️⃣ Mark Transit Node Candidates**
- Nodes involved in **boundary-crossing roads** are **potential transit nodes**.
- Saved in **`transit_nodes.json`**.

### **6️⃣ Filter Final Transit Nodes**
- Unimportant transit nodes are removed based on:
  - **Major roads are prioritized**.
  - **Nodes frequently used in shortest paths are kept**.
  - **Nearby transit nodes are merged**.
- Final transit nodes are saved in **`final_transit_nodes.json`**.

---

## 📊 File Structure
```
├── preprocessing.py         # Main preprocessing script
├── resources/              
│   ├── colorado_springs.graphml       # OSM road network
│   ├── colorado_springs_grid.geojson  # Generated square grid
├── preprocessing_output/   
│   ├── node_to_grid.json              # Nodes assigned to grid cells
│   ├── boundary_crossing_edges.json   # Roads crossing grid boundaries
│   ├── transit_nodes.json             # Transit node candidates
│   ├── final_transit_nodes.json       # Final transit nodes (optimized)
├── README.md                # Documentation
```
---
