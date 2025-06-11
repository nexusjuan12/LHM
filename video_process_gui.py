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
            
            progress(0.1, desc="Setting up processing...")
            
            # Copy video to working directory (like our successful test)
            working_video = Path("./1.mp4")
            shutil.copy2(video_file, working_video)
            
            progress(0.2, desc="Running pose estimation...")
            
            # Use the correct argument name: --output_path instead of --output_dir
            cmd = [
                "python", "./engine/pose_estimation/video2motion.py",
                "--video_path", "1.mp4",
                "--output_path", "./test_motion_output/"
            ]
            
            # Add half-body parameters if specified
            if is_half_body:
                cmd.extend(["--fitting_steps", "100", "0"])
            
            # Execute the command in the correct directory
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=".",
                timeout=600  # 10 minute timeout
            )
            
            progress(0.6, desc="Motion processing complete, organizing files...")
            
            # Check if processing was successful
            temp_output = Path("./test_motion_output/1")
            if not temp_output.exists() or not (temp_output / "smplx_params").exists():
                error_msg = f"❌ Motion processing failed for '{motion_name}'"
                if result.stderr:
                    error_msg += f"\n\nError details:\n{result.stderr}"
                if result.stdout:
                    error_msg += f"\n\nOutput:\n{result.stdout}"
                return error_msg, self.get_processed_motions()
            
            progress(0.8, desc="Setting up final motion folder...")
            
            # Create final motion directory
            final_motion_dir = self.output_base_dir / motion_name
            final_motion_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy processed motion data to final location
            if (final_motion_dir / "smplx_params").exists():
                shutil.rmtree(final_motion_dir / "smplx_params")
            shutil.copytree(temp_output / "smplx_params", final_motion_dir / "smplx_params")
            
            # Copy video files with correct names
            shutil.copy2(working_video, final_motion_dir / "input_video.mp4")
            shutil.copy2(working_video, final_motion_dir / "samurai_visualize.mp4")
            
            progress(0.9, desc="Cleaning up temporary files...")
            
            # Clean up temporary files
            if working_video.exists():
                working_video.unlink()
            if Path("./test_motion_output").exists():
                shutil.rmtree("./test_motion_output")
            
            progress(1.0, desc="✅ Motion extraction successful!")
            
            # Count motion frames
            motion_files = list((final_motion_dir / "smplx_params").glob("*.json"))
            frame_count = len(motion_files)
            
            return f"""✅ Successfully processed '{motion_name}'
📁 Motion saved to: {final_motion_dir / "smplx_params"}
🎬 Extracted {frame_count} motion frames
📺 Video files: input_video.mp4 & samurai_visualize.mp4

Ready to use in LHM! Your motion will appear as: {motion_name}""", self.get_processed_motions()
                
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
                    
                    ### 🔧 What this does:
                    
                    1. Copies your video to working directory
                    2. Runs pose estimation to extract motion
                    3. Creates motion folder with proper structure:
                       - `smplx_params/` (motion data)
                       - `input_video.mp4`
                       - `samurai_visualize.mp4`
                    4. Makes it available in LHM interface
                    """)
            
            status_output = gr.Textbox(
                label="Processing Status",
                lines=8,
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
            
            ### Expected Output Structure:
            ```
            train_data/motion_video/your_motion_name/
            ├── input_video.mp4           # Original video
            ├── samurai_visualize.mp4     # Copy for LHM
            └── smplx_params/             # Motion data folder
                ├── 00001.json            # Frame 1 motion
                ├── 00002.json            # Frame 2 motion
                └── ...                   # More frames
            ```
            
            ### Troubleshooting:
            - **No smplx_params folder**: Video processing failed, try shorter/clearer video
            - **Motion not in LHM**: Restart LHM app or check motion folder structure
            - **Processing timeout**: Video too long, try under 30 seconds
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
        share=True,            # Set to True if you want a public gradio link
        show_error=True
    )
