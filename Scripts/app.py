import helpers
from flask import Flask, render_template, request, redirect, url_for ,send_from_directory
import os
import threading
import time

# Initialize Flask app
app = Flask(__name__)
# Configure app settings for URL generation outside request context
app.config['APPLICATION_ROOT'] = '/'  # Usually left as '/'
app.config['PREFERRED_URL_SCHEME'] = 'http'  # Adjust if using HTTPS

# Define output folder for processed videos (changed to "outputs")
OUTPUT_FOLDER = 'outputs'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Define upload folder for videos
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#@app.route('/<path:filename>')
#def serve_video(filename):
    #return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


def change_extension(folder_path, old_ext=".avi", new_ext=".mp4"):
  """
  Changes the extension of all files in a folder from 'old_ext' to 'new_ext'.

  Args:
    folder_path: Path to the folder containing the files.
    old_ext: The current extension of the files (default: ".avi").
    new_ext: The desired new extension (default: ".mp4").
  """
  for filename in os.listdir(folder_path):
    if filename.endswith(old_ext):
      filepath = os.path.join(folder_path, filename)
      new_filename = os.path.splitext(filename)[0] + new_ext
      new_filepath = os.path.join(folder_path, new_filename)
      os.rename(filepath, new_filepath)
  
  print(f"Changed extensions of all {old_ext} files to {new_ext} in {folder_path}.")









@app.route('/outputs/<filename>')
def serve_video(filename):
  return send_from_directory(app.config['OUTPUT_FOLDER'],filename)

@app.route('/outputs/<filename>')
def display_inference(filename):
  # Construct the video path relative to the output folder
  video_url = url_for('serve_video', filename=filename)
  return render_template('inference.html', video_path=video_url)







@app.route('/', methods=['GET', 'POST'])
def upload_video():
  if request.method == 'POST':
    # Check if file is uploaded
    if 'video' not in request.files:
      return redirect(request.url)

    video_file = request.files['video']
    
    # Check if file is selected
    if video_file.filename == '':
      return redirect(request.url)

    # Check allowed file extension
    if video_file and allowed_file(video_file.filename):
      # Generate unique filename
      filename = f'{uuid.uuid4()}.{video_file.filename.rsplit(".", 1)[1]}'
      video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      # Output path within the "outputs" folder
      output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
      
      #output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'your_video.mp4')
      # Save video file
      video_file.save(video_path)

      # Get attacking side from form data
      attacking_side = request.form.get('side')  # Defaults to None if not selected


      success = helpers.prediction(video_path,output_path,attacking_side)
      
      #change_extension(app.config['OUTPUT_FOLDER']) 
      filename = filename.replace('.mp4','.avi')

      if success:
          time.sleep(2)
          print(output_path)
          return redirect(url_for('display_inference', filename=filename))

      # Redirect to loading page and start processing in background thread
      #def process_in_background(video_path, output_path, attacking_side):
        #with app.app_context():
          #success = helpers.prediction(video_path, output_path, attacking_side)
          #return success
         #if success:
            #print(output_path)
            #return redirect(url_for('display_inference', filename=output_path))
          #else:
            # Handle processing failure (e.g., display error message)
          #pass
      #process_in_background(video_path,output_path,attacking_side)
      #thread = threading.Thread(target=process_in_background, args=(video_path, output_path, attacking_side))
      #thread.start()
      #time.sleep(2)
      
      #return redirect(url_for('display_inference', filename=filename))
      #return render_template('loading.html')
      #if success:
  return render_template('upload.html')

#@app.route('/inference/<filename>')
#@app.route('/inference/<filename>')
#def display_inference(filename):
  # Construct the video path relative to the output folder
  #video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
  #video_path = url_for('serve_video', filename=filename)
  #return render_template('inference.html', video_path=video_path)

if __name__ == '__main__':
  # Import libraries for UUID generation (replace with your preferred method)
  import uuid
  app.run(host='0.0.0.0', port=5000, debug=True)
