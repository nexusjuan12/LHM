import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
import gradio as gr
import json

class VideoMotionProcessor:
    def __init__(self, output_base_dir="./train_data/motion_video"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True, parents=True)
        self.processing_status = {}
        
    def get_processed_motions(self):
        """Get list of all processed motion folders"""
        motion_folders = []
        if self.output_base_dir.exists():
            for folder in self.output_base_dir.iterdir():
                if folder.is_dir() and (folder / "smplx_params").exists():
                    motion_folders.append(str(folder / "smplx_params"))
        return motion_folders
    
    def process_video(self, video_file, motion_name, is_half_body=False, progress=gr.Progress()):
        """Process uploaded video and extract motion"""
        if video_file is None:
            return "❌ No video uploaded", self.get_processed_motions()
        
        try:
            # Create unique motion name if not provided
            if not motion_name:
                motion_name = f"motion_{int(time.time())}"
            
            # Clean motion name (remove special characters)
            motion_name = "".join(c for c in motion_name if c.isalnum() or c in (' ', '-', '_')).strip()
            motion_name = motion_name.replace(' ', '_')
            
            output_path = self.output_base_dir / motion_name
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Copy uploaded video to processing directory
            video_path = output_path / "input_video.mp4"
            shutil.copy2(video_file, video_path)
            
            progress(0.1, desc="Video uploaded, starting pose estimation...")
            
            # Build the processing command
            cmd = [
                "python", "./engine/pose_estimation/video2motion.py",
                "--video_path", str(video_path),
                "--output_path", str(output_path)
            ]
            
            # Add half-body parameters if specified
            if is_half_body:
                cmd.extend(["--fitting_steps", "100", "0"])
            
            progress(0.2, desc="Running pose estimation...")
            
            # Execute the command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=".",
                timeout=600  # 10 minute timeout
            )
            
            progress(0.8, desc="Processing complete, verifying output...")
            
            # Check if processing was successful
            smplx_params_path = output_path / "smplx_params"
            if smplx_params_path.exists() and any(smplx_params_path.iterdir()):
                progress(1.0, desc="✅ Motion extraction successful!")
                return f"✅ Successfully processed '{motion_name}'\n📁 Motion saved to: {smplx_params_path}", self.get_processed_motions()
            else:
                error_msg = f"❌ Processing failed for '{motion_name}'"
                if result.stderr:
                    error_msg += f"\n\nError details:\n{result.stderr}"
                return error_msg, self.get_processed_motions()
                
        except subprocess.TimeoutExpired:
            return f"❌ Processing timed out for '{motion_name}'. Try with a shorter video.", self.get_processed_motions()
        except Exception as e:
            return f"❌ Error processing '{motion_name}': {str(e)}", self.get_processed_motions()
    
    def delete_motion(self, motion_path):
        """Delete a processed motion"""
        try:
            if motion_path and Path(motion_path).exists():
                # Get parent directory (the motion folder)
                motion_folder = Path(motion_path).parent
                shutil.rmtree(motion_folder)
                return f"✅ Deleted motion: {motion_folder.name}", self.get_processed_motions()
            else:
                return "❌ Motion not found", self.get_processed_motions()
        except Exception as e:
            return f"❌ Error deleting motion: {str(e)}", self.get_processed_motions()

def create_gui():
    processor = VideoMotionProcessor()
    
    with gr.Blocks(title="LHM Motion Preprocessor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 LHM Video Motion Preprocessor
        
        Upload videos to extract motion parameters for use with LHM. 
        Processed motions will be automatically available in the main LHM gradio app.
        """)
        
        with gr.Tab("📤 Process New Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    video_input = gr.File(
                        label="Upload Video File",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                        type="filepath"
                    )
                    
                    motion_name = gr.Textbox(
                        label="Motion Name",
                        placeholder="e.g., 'dancing_clip' or 'workout_routine'",
                        info="Leave empty for auto-generated name"
                    )
                    
                    is_half_body = gr.Checkbox(
                        label="Half-body video",
                        info="Check this if your video only shows upper body/torso"
                    )
                    
                    process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📋 Tips for best results:
                    
                    - **Video quality**: Clear, well-lit videos work best
                    - **Duration**: 5-30 seconds recommended
                    - **Person visibility**: Full body should be visible (unless half-body mode)
                    - **Background**: Simple backgrounds work better
                    - **Motion**: Avoid very fast or erratic movements
                    """)
            
            status_output = gr.Textbox(
                label="Processing Status",
                lines=5,
                interactive=False
            )
        
        with gr.Tab("📁 Manage Processed Motions"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh List", size="sm")
                
            motion_list = gr.Dropdown(
                label="Processed Motions",
                choices=processor.get_processed_motions(),
                interactive=True,
                info="Select a motion to delete"
            )
            
            with gr.Row():
                delete_btn = gr.Button("🗑️ Delete Selected Motion", variant="stop")
            
            manage_status = gr.Textbox(
                label="Management Status",
                lines=2,
                interactive=False
            )
        
        with gr.Tab("ℹ️ Usage Instructions"):
            gr.Markdown("""
            ## How to use this preprocessor:
            
            ### Step 1: Process your video
            1. Upload your video file in the "Process New Video" tab
            2. Give it a descriptive name (optional)
            3. Check "Half-body video" if applicable
            4. Click "Process Video" and wait for completion
            
            ### Step 2: Use in main LHM app
            1. After processing completes, start the main LHM gradio app:
               ```bash
               python ./app_motion_ms.py
               ```
            2. Your processed motion will be available in the motion selection dropdown
            3. Upload a character image and generate your animation!
            
            ### Troubleshooting:
            - **Processing fails**: Try a shorter video or check video quality
            - **Motion not appearing**: Refresh the main LHM app or check the "Manage Processed Motions" tab
            - **Half-body videos**: Always check the "Half-body video" option for upper-body-only content
            
            ## 📁 Default Output Structure:

            The GUI saves processed motions to: `./train_data/motion_video/`
            
            This matches exactly where the main LHM gradio apps look for motions:
            ```
            train_data/motion_video/
            ├── your_motion_name_1/
            │   ├── input_video.mp4
            │   └── smplx_params/           # <- Motion data LHM uses
            ├── your_motion_name_2/
            │   ├── input_video.mp4  
            │   └── smplx_params/
            └── ...
            ```
            """)
        
        # Event handlers
        process_btn.click(
            fn=processor.process_video,
            inputs=[video_input, motion_name, is_half_body],
            outputs=[status_output, motion_list]
        )
        
        refresh_btn.click(
            fn=lambda: processor.get_processed_motions(),
            outputs=[motion_list]
        )
        
        delete_btn.click(
            fn=processor.delete_motion,
            inputs=[motion_list],
            outputs=[manage_status, motion_list]
        )
    
    return demo

if __name__ == "__main__":
    # Create the GUI
    demo = create_gui()
    
    # Launch with remote access
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7861,       # Different port from main LHM app
        share=False,            # Set to True if you want a public gradio link
        show_error=True
    )
