# ===================================================================
# DOWNLOAD ALL EXPERIMENTAL RESULTS & MODELS
# ===================================================================
import os
import zipfile
import shutil
from datetime import datetime
from google.colab import files

print("🚀 DOWNLOADING ALL EXPERIMENTAL RESULTS & MODELS")
print("=" * 60)

# Define paths (same as in your experimental suite)
DRIVE_ROOT = '/content/drive/MyDrive'
PROJECT_NAME = 'colab'
PROJECT_PATH = os.path.join(DRIVE_ROOT, PROJECT_NAME)
CHECKPOINT_DIR = os.path.join(PROJECT_PATH, 'checkpoints')
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')

# Create timestamp for download
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
download_package_name = f"experimental_results_{timestamp}.zip"

print(f"📦 Creating download package: {download_package_name}")

# Create temporary directory for packaging
temp_download_dir = f"/tmp/experimental_download_{timestamp}"
os.makedirs(temp_download_dir, exist_ok=True)

# Function to copy directory contents
def copy_directory_contents(src_dir, dst_dir, dir_name):
    """Copy directory contents and return file count"""
    if not os.path.exists(src_dir):
        print(f"   ⚠️  {dir_name} directory not found: {src_dir}")
        return 0
    
    dst_full_dir = os.path.join(dst_dir, dir_name)
    try:
        shutil.copytree(src_dir, dst_full_dir)
        file_count = sum([len(files) for r, d, files in os.walk(dst_full_dir)])
        print(f"   ✅ {dir_name}: {file_count} files copied")
        return file_count
    except Exception as e:
        print(f"   ❌ Error copying {dir_name}: {e}")
        return 0

# Copy checkpoints and results
total_files = 0
total_files += copy_directory_contents(CHECKPOINT_DIR, temp_download_dir, "checkpoints")
total_files += copy_directory_contents(RESULTS_PATH, temp_download_dir, "results")

if total_files == 0:
    print("❌ No files found to download!")
    print("   Make sure your experimental suite has completed successfully.")
else:
    print(f"\n📊 PACKAGE CONTENTS SUMMARY:")
    print(f"   Total files to download: {total_files}")
    
    # List key files for verification
    if os.path.exists(os.path.join(temp_download_dir, "checkpoints")):
        checkpoint_files = os.listdir(os.path.join(temp_download_dir, "checkpoints"))
        print(f"   Checkpoint files: {len(checkpoint_files)}")
        for f in checkpoint_files[:5]:  # Show first 5
            print(f"     - {f}")
        if len(checkpoint_files) > 5:
            print(f"     ... and {len(checkpoint_files) - 5} more")
    
    if os.path.exists(os.path.join(temp_download_dir, "results")):
        results_dirs = os.listdir(os.path.join(temp_download_dir, "results"))
        print(f"   Result directories: {len(results_dirs)}")
        for d in results_dirs[:3]:  # Show first 3
            print(f"     - {d}")
        if len(results_dirs) > 3:
            print(f"     ... and {len(results_dirs) - 3} more")

    # Create ZIP file
    print(f"\n🗜️  Creating ZIP archive...")
    zip_path = f"/tmp/{download_package_name}"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_download_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Create relative path for ZIP
                arcname = os.path.relpath(file_path, temp_download_dir)
                zipf.write(file_path, arcname)
    
    # Get ZIP file size
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   ✅ ZIP created: {zip_size_mb:.2f} MB")
    
    # Download the ZIP file
    print(f"\n⬇️  STARTING DOWNLOAD...")
    print(f"   File: {download_package_name}")
    print(f"   Size: {zip_size_mb:.2f} MB")
    print(f"   Contents: {total_files} files")
    
    try:
        files.download(zip_path)
        print(f"   ✅ Download initiated successfully!")
        print(f"\n📁 Your browser should start downloading: {download_package_name}")
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print(f"   💡 Try running this cell again, or check browser popup blockers")

    # Cleanup temporary files
    print(f"\n🧹 Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_download_dir)
        os.remove(zip_path)
        print("   ✅ Cleanup complete")
    except:
        print("   ⚠️  Some temporary files may remain")

print(f"\n🎯 DOWNLOAD SUMMARY:")
print(f"   Package: {download_package_name}")
print(f"   Total files: {total_files}")
print(f"   Status: {'✅ Ready for download' if total_files > 0 else '❌ No files found'}")

if total_files > 0:
    print(f"\n📋 WHAT YOU'RE DOWNLOADING:")
    print(f"   📦 Checkpoints: All your best model files")
    print(f"   📊 Results: Complete experimental metadata & histories")
    print(f"   🏆 Best Models: Easy-access copies of winning configurations")
    print(f"   📄 Summaries: JSON & text summaries of all experiments")
    print(f"\n💡 After download, extract the ZIP to access your models!") 