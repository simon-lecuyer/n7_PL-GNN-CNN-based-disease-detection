"""
GNN Data Preparation for WaterberryFarms Disease Prediction

This script extracts and structures graph data (nodes, edges, temporal features) 
from WaterberryFarms simulator for training a Graph Neural Network to predict disease propagation.

Key Features:
- Converts 2D grid agricultural field to graph representation
- 4-nearest neighbor connectivity (up, down, left, right) : for now, later we will add more complex connectivity
- Temporal edges connecting nodes across time steps
- Node features: position, crop type, observation mask, disease state, infection intensity
"""

import sys
import os
# Add parent directory to path to import WaterberryFarms modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# WaterberryFarms imports
from environment import EpidemicSpreadEnvironment
from water_berry_farm import FarmGeometry


class GraphDataGenerator:
    """
    Generates graph-structured data from WaterberryFarms simulator
    for GNN-based disease prediction.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        temporal_window: int = 5,
        prediction_horizon: int = 1,
        seed: int = 42
    ):
        """
        Initialize the graph data generator.
        
        Args:
            width: Field width (number of cells)
            height: Field height (number of cells)
            temporal_window: Number of past time steps to include (L)
            prediction_horizon: How many time steps ahead to predict (k)
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.temporal_window = temporal_window
        self.prediction_horizon = prediction_horizon
        self.seed = seed
        self.num_nodes = width * height
        
        print(f"Initialized GraphDataGenerator: {width}x{height} grid = {self.num_nodes} nodes")
        
    def _cell_to_node_id(self, x: int, y: int) -> int:
        """Convert 2D grid coordinates to 1D node ID."""
        return x * self.height + y
    
    def _node_id_to_cell(self, node_id: int) -> Tuple[int, int]:
        """Convert 1D node ID to 2D grid coordinates."""
        x = node_id // self.height
        y = node_id % self.height
        return x, y
    
    def build_spatial_edges_4nn(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build spatial edges using 4-nearest neighbors connectivity.
        Each cell connects to its up, down, left, right neighbors.
        
        Returns:
            edge_index: Array of shape (2, num_edges) with source and target node IDs
            edge_weights: Array of shape (num_edges,) with edge weights (all 1.0 for now)
        """
        print("Building 4-nearest neighbor spatial edges")
        
        edges = []
        
        for x in range(self.width):
            for y in range(self.height):
                node_id = self._cell_to_node_id(x, y)
                
                # 4 directions: up, down, left, right
                neighbors = [
                    (x-1, y),  # left
                    (x+1, y),  # right
                    (x, y-1),  # down
                    (x, y+1),  # up
                ]
                
                for nx, ny in neighbors:
                    # Check if neighbor is within bounds
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        neighbor_id = self._cell_to_node_id(nx, ny)
                        edges.append([node_id, neighbor_id])
        
        edges = np.array(edges, dtype=np.int64).T  # Shape: (2, num_edges)
        edge_weights = np.ones(edges.shape[1], dtype=np.float32)
        
        print(f"Created {edges.shape[1]} spatial edges")
        return edges, edge_weights
    
    def build_temporal_edges(self, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build temporal edges connecting each node to itself at previous time step.
        Node i at time t connects to node i at time t-1.
        
        Args:
            time_steps: Total number of time steps in the sequence
            
        Returns:
            edge_index: Array of shape (2, num_temporal_edges)
            edge_weights: Array of shape (num_temporal_edges,)
        """
        print(f"Building temporal edges for {time_steps} time steps...")
        
        edges = []
        
        for t in range(1, time_steps):
            for node_id in range(self.num_nodes):
                # Node at time t connects to same node at time t-1
                # Encoding: node_at_t = node_id + t * num_nodes
                source = node_id + (t-1) * self.num_nodes  # time t-1
                target = node_id + t * self.num_nodes      # time t
                edges.append([source, target])
        
        edges = np.array(edges, dtype=np.int64).T
        edge_weights = np.ones(edges.shape[1], dtype=np.float32)
        
        print(f"Created {edges.shape[1]} temporal edges")
        return edges, edge_weights
    
    def extract_node_features(
        self,
        environment: EpidemicSpreadEnvironment,
        crop_type_map: np.ndarray,
        observation_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract node features from environment state.
        
        Features per node:
        - Positional encoding: normalized (x, y) coordinates [2 features]
        - Crop type: one-hot encoded [3 features: tomato, strawberry, unplanted]
        - Observation mask: binary indicator [1 feature]
        - Disease state: encoded status [-2, -1, 0, or >0] [1 feature]
        - Value field: infection intensity [0.0 or 0.5] [1 feature]
        
        Total: 8 features per node
        
        Args:
            environment: EpidemicSpreadEnvironment instance
            crop_type_map: Array (width, height) with crop types (0=unplanted, 1=crop1, 2=crop2)
            observation_mask: Array (width, height) with binary mask (1=observed, 0=not observed)
            
        Returns:
            node_features: Array of shape (num_nodes, num_features)
        """
        features_list = []
        
        for x in range(self.width):
            for y in range(self.height):
                node_features = []
                
                # 1. Positional encoding (normalized)
                pos_x = x / self.width
                pos_y = y / self.height
                node_features.extend([pos_x, pos_y])
                
                # 2. Crop type (one-hot encoded: 3 categories)
                crop_type = int(crop_type_map[x, y])
                crop_one_hot = [0, 0, 0]
                if 0 <= crop_type < 3:
                    crop_one_hot[crop_type] = 1
                node_features.extend(crop_one_hot)
                
                # 3. Observation mask
                obs_mask = float(observation_mask[x, y])
                node_features.append(obs_mask)
                
                # 4. Disease state (from status field)
                disease_state = float(environment.status[x, y])
                node_features.append(disease_state)
                
                # 5. Value field (infection intensity)
                value = float(environment.value[x, y])
                node_features.append(value)
                
                features_list.append(node_features)
        
        node_features = np.array(features_list, dtype=np.float32)
        print(f"Extracted node features: shape {node_features.shape}")
        return node_features
    
    def extract_targets(
        self,
        environment: EpidemicSpreadEnvironment
    ) -> np.ndarray:
        """
        Extract target values for prediction.
        Converts disease state to continuous infection probability [0, 1].
        
        Mapping:
        - status = -2 (immune) -> 0.0
        - status = -1 (recovered) -> 0.0
        - status = 0 (susceptible) -> 0.0
        - status > 0 (infected) -> 1.0 (can be refined to intensity)
        
        Args:
            environment: EpidemicSpreadEnvironment instance
            
        Returns:
            targets: Array of shape (num_nodes,) with values in [0, 1]
        """
        targets = []
        
        for x in range(self.width):
            for y in range(self.height):
                status = environment.status[x, y]
                
                # Convert to continuous infection probability
                if status > 0:  # Infected
                    # Option 1: Binary (0 or 1)
                    infection_prob = 1.0
                    
                    # Option 2: Intensity based on remaining infection days
                    # infection_prob = status / environment.infection_duration
                else:
                    infection_prob = 0.0
                
                targets.append(infection_prob)
        
        targets = np.array(targets, dtype=np.float32)
        return targets
    
    def generate_simulation_sequence(
        self,
        farm_geometry: Optional[FarmGeometry] = None,
        num_time_steps: int = 20,
        observation_rate: float = 0.3,
        p_transmission: float = 0.2,
        infection_duration: int = 5
    ) -> Dict:
        """
        Generate a complete simulation sequence with observations.
        
        Args:
            farm_geometry: FarmGeometry instance (if None, creates simple uniform field)
            num_time_steps: Number of time steps to simulate
            observation_rate: Fraction of cells observed by drone at each time step
            p_transmission: Disease transmission probability
            infection_duration: Duration of infection in time steps
            
        Returns:
            data_dict: Dictionary containing:
                - 'node_features': List of feature arrays for each time step
                - 'observation_masks': List of observation masks
                - 'targets': List of target arrays
                - 'edge_index_spatial': Spatial edges
                - 'edge_weights_spatial': Spatial edge weights
                - 'metadata': Simulation parameters
        """
        print(f"Generating simulation sequence: {num_time_steps} time steps")
        
        # Create or use provided farm geometry
        if farm_geometry is None:
            # Simple uniform field (no specific crop types)
            crop_type_map = np.zeros((self.width, self.height), dtype=np.int32)
            immunity_mask = None
        else:
            farm_geometry.create_type_map()
            # Resize type_map to match our grid dimensions if needed
            if farm_geometry.type_map.shape != (self.width, self.height):
                # Crop or pad to match dimensions
                crop_type_map = np.zeros((self.width, self.height), dtype=np.int32)
                min_w = min(farm_geometry.type_map.shape[0], self.width)
                min_h = min(farm_geometry.type_map.shape[1], self.height)
                crop_type_map[:min_w, :min_h] = farm_geometry.type_map[:min_w, :min_h]
            else:
                crop_type_map = farm_geometry.type_map
            
            # Create immunity mask: -2 for unplanted areas, 0 for others
            immunity_mask = np.zeros((self.width, self.height))
            immunity_mask[crop_type_map == 0] = -2  # Unplanted areas are immune
        
        # Initialize environment
        env = EpidemicSpreadEnvironment(
            name="disease",
            width=self.width,
            height=self.height,
            seed=self.seed,
            p_transmission=p_transmission,
            infection_duration=infection_duration,
            immunity_mask=immunity_mask
        )
        
        # Build spatial edges (static across time)
        edge_index_spatial, edge_weights_spatial = self.build_spatial_edges_4nn()
        
        # Storage for time series data
        node_features_sequence = []
        observation_masks_sequence = []
        targets_sequence = []
        
        # Simulate and collect data
        for t in range(num_time_steps):
            print(f"Time step {t}/{num_time_steps}")
            
            # Generate observation mask (which cells are observed by drone)
            observation_mask = np.random.rand(self.width, self.height) < observation_rate
            observation_mask = observation_mask.astype(np.float32)
            
            # Extract features
            node_features = self.extract_node_features(env, crop_type_map, observation_mask)
            node_features_sequence.append(node_features)
            observation_masks_sequence.append(observation_mask)
            
            # Environment proceeds
            env.proceed(delta_t=1.0)
            
            # Extract targets (state at t+k where k=prediction_horizon)
            if t + self.prediction_horizon < num_time_steps:
                # Need to peek ahead - for now, just use current state
                # In practice, you'd run environment ahead or store future states
                targets = self.extract_targets(env)
                targets_sequence.append(targets)
        
        # Compile data dictionary
        data_dict = {
            'node_features_sequence': node_features_sequence,
            'observation_masks_sequence': observation_masks_sequence,
            'targets_sequence': targets_sequence,
            'edge_index_spatial': edge_index_spatial,
            'edge_weights_spatial': edge_weights_spatial,
            'crop_type_map': crop_type_map,
            'metadata': {
                'width': self.width,
                'height': self.height,
                'num_nodes': self.num_nodes,
                'num_time_steps': num_time_steps,
                'temporal_window': self.temporal_window,
                'prediction_horizon': self.prediction_horizon,
                'observation_rate': observation_rate,
                'p_transmission': p_transmission,
                'infection_duration': infection_duration,
                'seed': self.seed,
                'num_features_per_node': 8,
                'feature_description': [
                    'pos_x', 'pos_y',
                    'crop_type_0', 'crop_type_1', 'crop_type_2',
                    'observation_mask',
                    'disease_state',
                    'value_field'
                ]
            }
        }
        
        print("Simulation sequence generation complete")
        return data_dict
    
    def save_dataset(self, data_dict: Dict, output_path: str):
        """Save dataset to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Dataset saved to {output_path}")
    
    def load_dataset(self, input_path: str) -> Dict:
        """Load dataset from disk."""
        with open(input_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        print(f"Dataset loaded from {input_path}")
        return data_dict


def create_simple_farm_geometry(width: int, height: int) -> FarmGeometry:
    """
    Create a simple farm geometry for testing.
    
    Layout:
    - Left half: tomato crop (type 1)
    - Right half: strawberry crop (type 2)
    """
    geometry = FarmGeometry()
    
    mid = width // 2
    
    # Tomato patch (left)
    tomato_area = np.array([
        [0, 0],
        [mid, 0],
        [mid, height],
        [0, height]
    ])
    geometry.add_patch("tomato_field", "tomato", tomato_area, "red")
    
    # Strawberry patch (right)
    strawberry_area = np.array([
        [mid, 0],
        [width, 0],
        [width, height],
        [mid, height]
    ])
    geometry.add_patch("strawberry_field", "strawberry", strawberry_area, "pink")
    
    return geometry


def visualize_graph_snapshot(data_dict: Dict, time_step: int = 0):
    """
    Visualize a snapshot of the graph at a given time step.
    
    Args:
        data_dict: Data dictionary from generate_simulation_sequence
        time_step: Which time step to visualize
    """
    import matplotlib.pyplot as plt
    
    width = data_dict['metadata']['width']
    height = data_dict['metadata']['height']
    
    # Get features at time step
    features = data_dict['node_features_sequence'][time_step]
    obs_mask = data_dict['observation_masks_sequence'][time_step]
    
    # Reshape for visualization
    infection_intensity = features[:, -1].reshape(width, height)
    observation = obs_mask.reshape(width, height)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot infection intensity
    im1 = axes[0].imshow(infection_intensity.T, origin='lower', cmap='Reds')
    axes[0].set_title(f'Infection Intensity (t={time_step})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot observation mask
    im2 = axes[1].imshow(observation.T, origin='lower', cmap='Greys')
    axes[1].set_title(f'Observation Mask (t={time_step})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot graph connectivity (show some edges)
    edges = data_dict['edge_index_spatial']
    axes[2].set_xlim(0, width)
    axes[2].set_ylim(0, height)
    axes[2].set_aspect('equal')
    
    # Sample edges for visualization (otherwise too many)
    sample_edges = edges[:, ::100]  # Show every 100th edge
    
    for i in range(sample_edges.shape[1]):
        src, tgt = sample_edges[:, i]
        src_x, src_y = src // height, src % height
        tgt_x, tgt_y = tgt // height, tgt % height
        axes[2].plot([src_x, tgt_x], [src_y, tgt_y], 'b-', alpha=0.3, linewidth=0.5)
    
    axes[2].set_title('Graph Connectivity (sampled)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('graph_snapshot.png', dpi=150)
    print("Visualization saved to graph_snapshot.png")
    plt.show()


def main():
    """
    Main function to demonstrate dataset generation.
    """
    print("=" * 80)
    print("GNN Data Preparation - WaterberryFarms Disease Prediction")
    print("=" * 80)
    
    # Configuration
    FIELD_WIDTH = 30
    FIELD_HEIGHT = 30
    NUM_TIME_STEPS = 30
    TEMPORAL_WINDOW = 5
    PREDICTION_HORIZON = 1
    OBSERVATION_RATE = 0.3
    SEED = 42
    
    # Initialize generator
    generator = GraphDataGenerator(
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT,
        temporal_window=TEMPORAL_WINDOW,
        prediction_horizon=PREDICTION_HORIZON,
        seed=SEED
    )
    
    # Create simple farm geometry
    farm_geometry = create_simple_farm_geometry(FIELD_WIDTH, FIELD_HEIGHT)
    
    # Generate simulation data
    data_dict = generator.generate_simulation_sequence(
        farm_geometry=farm_geometry,
        num_time_steps=NUM_TIME_STEPS,
        observation_rate=OBSERVATION_RATE,
        p_transmission=0.2,
        infection_duration=5
    )
    
    # Print dataset statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics:")
    print("=" * 80)
    print(f"Grid size: {FIELD_WIDTH} x {FIELD_HEIGHT}")
    print(f"Number of nodes: {generator.num_nodes}")
    print(f"Number of time steps: {NUM_TIME_STEPS}")
    print(f"Number of spatial edges: {data_dict['edge_index_spatial'].shape[1]}")
    print(f"Features per node: {data_dict['metadata']['num_features_per_node']}")
    print(f"Feature names: {data_dict['metadata']['feature_description']}")
    print(f"Observation rate: {OBSERVATION_RATE * 100:.1f}%")
    
    # Calculate average infection rate
    avg_infection = []
    for features in data_dict['node_features_sequence']:
        infection_values = features[:, -1]  # Last feature is value field
        avg_infection.append(infection_values.mean())
    print(f"Average infection rate over time: {np.mean(avg_infection):.3f}")
    
    # Save dataset
    output_dir = Path(__file__).parent / "data"
    output_path = output_dir / f"disease_graph_dataset_{FIELD_WIDTH}x{FIELD_HEIGHT}_t{NUM_TIME_STEPS}.pkl"
    generator.save_dataset(data_dict, str(output_path))
    
    # Visualize a snapshot
    print("\nGenerating visualization :")
    visualize_graph_snapshot(data_dict, time_step=10)

if __name__ == "__main__":
    main()
