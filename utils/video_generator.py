"""
Video Generator
Generates videos from frames and scene graphs, similar to REFLECT's video generation
"""

import os
import numpy as np
from typing import List, Optional, Dict
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class VideoGenerator:
    """Generates videos from frames and scene graphs"""
    
    def __init__(self, output_dir: str = "output/videos"):
        """
        Initialize video generator
        
        Args:
            output_dir: Directory to save videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_frame_with_annotations(self,
                                     frame: np.ndarray,
                                     scene_graph,
                                     step_idx: int,
                                     action_info: Optional[Dict] = None) -> np.ndarray:
        """
        Create an annotated frame with scene graph information
        
        Args:
            frame: RGB frame (H x W x 3)
            scene_graph: SceneGraph object
            step_idx: Frame index
            action_info: Optional action information dict
            
        Returns:
            Annotated frame as numpy array
        """
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            annotated_frame = frame.copy()
        else:
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main frame
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(annotated_frame)
        ax1.set_title(f'Frame {step_idx}', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Annotate objects on frame
        colors = plt.cm.tab10(np.linspace(0, 1, len(scene_graph.nodes)))
        for i, node in enumerate(scene_graph.nodes):
            if hasattr(node, 'attributes') and node.attributes.get('bbox2d') is not None:
                bbox = node.attributes['bbox2d']
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    rect = mpatches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor=colors[i], facecolor='none'
                    )
                    ax1.add_patch(rect)
                    ax1.text(x1, y1-5, node.get_name(),
                            fontsize=10, color=colors[i], fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Scene graph visualization
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_scene_graph(ax2, scene_graph, colors)
        ax2.set_title('Scene Graph', fontsize=12, fontweight='bold')
        
        # Action information
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        info_text = f"Step: {step_idx}\n"
        if action_info:
            info_text += f"Action: {action_info.get('type', 'N/A')}\n"
            info_text += f"Target: {action_info.get('target', 'N/A')}\n"
            if action_info.get('status'):
                info_text += f"Status: {action_info.get('status')}\n"
        info_text += f"\nNodes: {len(scene_graph.nodes)}\n"
        info_text += f"Edges: {len(scene_graph.edges)}"
        ax3.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Convert figure to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return buf
    
    def _draw_scene_graph(self, ax, scene_graph, colors):
        """Draw scene graph structure"""
        import networkx as nx
        
        G = nx.DiGraph()
        node_labels = {}
        
        # Add nodes
        for i, node in enumerate(scene_graph.nodes):
            label = f"{node.get_name()}\n({node.object_type})"
            G.add_node(node.name, label=label)
            node_labels[node.name] = label
        
        # Add edges
        for (start_name, end_name), edge in scene_graph.edges.items():
            G.add_edge(start_name, end_name, label=edge.edge_type)
        
        # Create layout
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=1.5, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                                 node_size=2000, alpha=0.9)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                                  font_size=8, font_weight='bold')
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                                  arrows=True, arrowsize=15, width=1.5)
            
            # Edge labels
            edge_labels = {}
            for (start, end), edge_data in G.edges.items():
                edge_labels[(start, end)] = edge_data.get('label', '')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
    
    def generate_video(self,
                      frames: List[np.ndarray],
                      scene_graphs: List,
                      step_indices: List[int],
                      action_infos: Optional[List[Dict]] = None,
                      output_filename: str = "demo_video.mp4",
                      fps: int = 5) -> str:
        """
        Generate video from frames and scene graphs
        
        Args:
            frames: List of RGB frames (each H x W x 3)
            scene_graphs: List of SceneGraph objects
            step_indices: List of step indices
            action_infos: Optional list of action information dicts
            output_filename: Output video filename
            fps: Frames per second
            
        Returns:
            Path to generated video
        """
        if len(frames) != len(scene_graphs):
            raise ValueError(f"Frames ({len(frames)}) and scene graphs ({len(scene_graphs)}) must have same length")
        
        if action_infos is None:
            action_infos = [None] * len(frames)
        
        print(f"\n{'='*60}")
        print(f"Generating Video: {output_filename}")
        print(f"{'='*60}")
        print(f"Total frames: {len(frames)}")
        print(f"FPS: {fps}")
        print(f"Output directory: {self.output_dir}")
        
        # Create annotated frames
        annotated_frames = []
        for i, (frame, scene_graph, step_idx) in enumerate(zip(frames, scene_graphs, step_indices)):
            print(f"Processing frame {i+1}/{len(frames)} (step {step_idx})...")
            action_info = action_infos[i] if i < len(action_infos) else None
            annotated_frame = self.create_frame_with_annotations(
                frame, scene_graph, step_idx, action_info
            )
            annotated_frames.append(annotated_frame)
        
        # Save video
        output_path = self.output_dir / output_filename
        print(f"\nSaving video to: {output_path}")
        
        # Get frame dimensions
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in annotated_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✅ Video saved successfully: {output_path}")
        
        return str(output_path)
    
    def generate_video_from_events(self,
                                   events: List,
                                   scene_graphs: List,
                                   step_indices: List[int],
                                   action_infos: Optional[List[Dict]] = None,
                                   output_filename: str = "demo_video.mp4",
                                   fps: int = 5) -> str:
        """
        Generate video from AI2THOR events
        
        Args:
            events: List of AI2THOR event objects (can be None for mock events)
            scene_graphs: List of SceneGraph objects
            step_indices: List of step indices
            action_infos: Optional list of action information dicts
            output_filename: Output video filename
            fps: Frames per second
            
        Returns:
            Path to generated video
        """
        # Extract frames from events
        frames = []
        for i, event in enumerate(events):
            if event is not None and hasattr(event, 'frame'):
                frame = event.frame
                # Convert to RGB if needed
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:  # RGBA
                        frame = frame[:, :, :3]
                    frames.append(frame)
                else:
                    # Grayscale or other format
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                # Create mock frame with text annotation
                mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(mock_frame, f"Step {step_indices[i] if i < len(step_indices) else i}", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                frames.append(mock_frame)
        
        return self.generate_video(frames, scene_graphs, step_indices, action_infos, output_filename, fps)

