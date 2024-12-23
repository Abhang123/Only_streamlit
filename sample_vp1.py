import cv2
import numpy as np
import streamlit as st
import tempfile 
from PIL import Image
from streamlit_lottie import st_lottie
import json

# Streamlit UI setup
st.title("Yellow Split Peas Quality Analysis")

with open("cv_animation.json") as source:
            animation = json.load(source)

st_lottie(animation, width = 800)


option = st.selectbox(
    "WHich daal do you want to analyze?",
    ("Toor Daal","Moong Daal", "Harbhara Daal"),
)

st.write("\n")
st.write("\n")

if option == "Toor Daal":

    photo1 = st.camera_input("Take first photo of toor daal")
    st.write("\n")
    st.write("\n")
    if photo1:
        st.write("Image 1 of toor daal captured successfully!")
        st.write("\n")
        photo2 = st.camera_input("Take second photo of toor daal.")
        st.write("\n")
        st.write("\n")
        if photo2:
            st.write("Image 2 of toor daal captured successfully!")
            st.write("\n")
            photo3 = st.camera_input("Take third photo of toor daal.")
            if photo3:
                st.write("Image 3 captured successfully!")
            st.write("\n")
            st.write("\n")
    else:
        st.write("Please take first photo of toor daal.")

    video_upload = st.file_uploader("Upload a video file of Toor Daal", type=["mp4", "avi", "mov"])
    st.write("\n")
    st.write("\n")

    if video_upload is not None:
        
        st.title("Toor Daal Quality Assessment")
        st.write("\n")
        st.write("\n")
        
        inputNumber = st.number_input("Enter your required moisture content (in percentage) : ")
        st.write("\n")
        st.write("\n")
        
        if inputNumber:
            # Save the uploaded video temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(video_upload.read())
            video_path = temp_file.name  

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Error: Could not open the uploaded video.")
                st.stop()

            st.success("Toor Daal Video uploaded successfully! Processing...")

            minimum_threshold_mature_percentage = 70

            mature_lower = np.array([78, 148, 93])
            mature_upper = np.array([180, 231, 255])

            immature_lower = np.array([20, 70, 70])
            immature_upper = np.array([30, 200, 150])

            aged_lower = np.array([35, 100, 100])
            aged_upper = np.array([50, 255, 255])

            moist_lower = np.array([22, 120, 100])
            moist_upper = np.array([32, 200, 200])
            dry_lower = np.array([25, 150, 180])
            dry_upper = np.array([35, 255, 255])


            # Placeholders for dynamic updates
            frame_placeholder = st.empty()
            moist_percent_placeholder = st.empty()
            mature_percent_placeholder = st.empty()
            result_placeholder = st.empty()

            # Initialize variables for aggregation
            total_mature_area = 0
            total_immature_area = 0
            total_aged_area = 0
            total_moist_area = 0
            total_dry_area = 0
            total_area = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Process video frames
            step = 10  # Process every 10th frame
            for frame_idx in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

            # Resize frame for better performance
            frame_resized = cv2.resize(frame, (640, 360))


            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    
            mature_mask = cv2.inRange(hsv, mature_lower, mature_upper)
            immature_mask = cv2.inRange(hsv, immature_lower, immature_upper)
            aged_mask = cv2.inRange(hsv, aged_lower, aged_upper)

            moist_mask = cv2.inRange(hsv, moist_lower, moist_upper)
            dry_mask = cv2.inRange(hsv, dry_lower, dry_upper)

            mature_area = cv2.countNonZero(mature_mask)
            immature_area = cv2.countNonZero(immature_mask)
            aged_area = cv2.countNonZero(aged_mask)
            moist_area = cv2.countNonZero(moist_mask)
            dry_area = cv2.countNonZero(dry_mask)

            frame_total_area = mature_area + immature_area + aged_area

            total_mature_area += mature_area
            total_immature_area += immature_area
            total_aged_area += aged_area
            total_moist_area += moist_area
            total_dry_area += dry_area
            total_area += frame_total_area

            moist_percent = (moist_area / (moist_area + dry_area)) * 100 if (moist_area + dry_area) > 0 else 0
            dry_percent = (dry_area / (moist_area + dry_area)) * 100 if (moist_area + dry_area) > 0 else 0

            # moist_percent_placeholder.markdown(f"### Moist Percent: {moist_percent:.2f}%")
            frame_placeholder.image(frame_resized, channels="BGR", use_column_width=True)

            mature_percent = (total_mature_area / total_area) * 100 if total_area > 0 else 0
            immature_percent = (total_immature_area / total_area) * 100 if total_area > 0 else 0
            aged_percent = (total_aged_area / total_area) * 100 if total_area > 0 else 0

            #~~~~~~~~~~~~~~~~~~~~~~


        # ~~~~~~~~~~~~~~~ Black Dot Analysis ~~~~~~~~~~~~~~~~~~~

            blur = cv2.medianBlur(frame, 5)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            black_dots = [c for c in contours if cv2.contourArea(c) > 5]

            frame_black_dots = frame.copy()
            cv2.drawContours(frame_black_dots, black_dots, -1, (0, 255, 0), 2)

            frame_black_dots_rgb = cv2.cvtColor(frame_black_dots, cv2.COLOR_BGR2RGB)


            #~~~~~~~~~~~~~~~~~ Faint Black Dots ~~~~~~~~~~~~~~~~~

            # Faint Black Dot Analysis
            blur = cv2.medianBlur(frame, 5)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            faint_black_dots = [c for c in contours if cv2.contourArea(c) > 5]

            frame_faint_black_dots = frame.copy()
            cv2.drawContours(frame_faint_black_dots, faint_black_dots, -1, (0, 255, 0), 2)

            frame_faint_black_dots_rgb = cv2.cvtColor(frame_faint_black_dots, cv2.COLOR_BGR2RGB)



            # -------------------------- Photo 1 Analysis -------------------- #

            image1 = Image.open(photo1)
            image_np1 = np.array(image1)

            # Convert to NumPy array for OpenCV
            resized_image1 = cv2.resize(image_np1, (500, 500))

            hsv1 = cv2.cvtColor(resized_image1, cv2.COLOR_RGB2HSV)

            # mature_lower = np.array([25, 80, 80])
            # mature_upper = np.array([35, 255, 255])
            mature_lower1 = np.array([20, 70, 70])  # Lower saturation and value for darker/lighter yellows
            mature_upper1 = np.array([40, 255, 255])  # Higher hue to capture yellow-orange tones


            immature_lower1 = np.array([20, 70, 70]) 
            immature_upper1 = np.array([30, 200, 150])

            aged_lower1 = np.array([35, 100, 100])
            aged_upper1 = np.array([50, 255, 255])


            mature_mask1 = cv2.inRange(hsv1, mature_lower1, mature_upper1)
            immature_mask1 = cv2.inRange(hsv1, immature_lower1, immature_upper1)
            aged_mask1 = cv2.inRange(hsv1, aged_lower1, aged_upper1)
            

            mature_area1 = cv2.countNonZero(mature_mask1)
            immature_area1 = cv2.countNonZero(immature_mask1)
            aged_area1 = cv2.countNonZero(aged_mask1)
            total_area1 = mature_area1 + aged_area1 + immature_area1


            moist_lower1 = np.array([22, 120, 100])
            moist_upper1 = np.array([32, 200, 200])
            
            dry_lower1 = np.array([25, 150, 180])
            dry_upper1 = np.array([35, 255, 255])
        

            moist_mask1 = cv2.inRange(hsv, moist_lower, moist_upper)
            dry_mask1 = cv2.inRange(hsv, dry_lower, dry_upper)

            moist_area1 = cv2.countNonZero(moist_mask1)
            dry_area1 = cv2.countNonZero(dry_mask1)

            total_moist_area1 = moist_area1 + dry_area1

            mature_percent1 = (mature_area1 / total_area1) * 100 if total_area1 > 0 else 0
            aged_percent1 = (aged_area1 / total_area1) * 100 if total_area1 > 0 else 0
            immature_percent1 = (immature_area1 / total_area1) * 100 if total_area1 > 0 else 0

            moist_percent1 = (moist_area1 / total_moist_area1) * 10 if total_moist_area1 > 0 else 0
            dry_percent1 = (dry_area1 / total_moist_area1) * 100 if total_moist_area1 > 0 else 0        


            # ---------------------------- Photo 2 Analysis ---------------------- #


            image2 = Image.open(photo2)
            image_np2 = np.array(image2)

            # Convert to NumPy array for OpenCV
            resized_image2 = cv2.resize(image_np2, (500, 500))

            hsv2 = cv2.cvtColor(resized_image2, cv2.COLOR_RGB2HSV)


            mature_mask2 = cv2.inRange(hsv2, mature_lower1, mature_upper1)
            immature_mask2 = cv2.inRange(hsv2, immature_lower1, immature_upper1)
            aged_mask2 = cv2.inRange(hsv2, aged_lower1, aged_upper1)
            

            mature_area2 = cv2.countNonZero(mature_mask2)
            immature_area2 = cv2.countNonZero(immature_mask2)
            aged_area2 = cv2.countNonZero(aged_mask2)
            total_area2 = mature_area2 + aged_area2 + immature_area2
        

            moist_mask2 = cv2.inRange(hsv2, moist_lower1, moist_upper1)
            dry_mask2 = cv2.inRange(hsv2, dry_lower1, dry_upper1)

            moist_area2 = cv2.countNonZero(moist_mask2)
            dry_area2 = cv2.countNonZero(dry_mask2)

            total_moist_area2 = moist_area2 + dry_area2

            mature_percent2 = (mature_area2 / total_area2) * 100 if total_area2 > 0 else 0
            aged_percent2 = (aged_area2 / total_area2) * 100 if total_area2 > 0 else 0
            immature_percent2 = (immature_area2 / total_area2) * 100 if total_area2 > 0 else 0

            moist_percent2 = (moist_area2 / total_moist_area2) * 10 if total_moist_area2 > 0 else 0
            dry_percent2 = (dry_area2 / total_moist_area2) * 100 if total_moist_area2 > 0 else 0        



            # ---------------------- Photo 3 Analysis ------------------#


            image3 = Image.open(photo3)
            image_np3 = np.array(image3)

            # Convert to NumPy array for OpenCV
            resized_image3 = cv2.resize(image_np3, (500, 500))

            hsv3 = cv2.cvtColor(resized_image3, cv2.COLOR_RGB2HSV)

            mature_mask3 = cv2.inRange(hsv3, mature_lower1, mature_upper1)
            immature_mask3 = cv2.inRange(hsv3, immature_lower1, immature_upper1)
            aged_mask3 = cv2.inRange(hsv3, aged_lower1, aged_upper1)
            

            mature_area3 = cv2.countNonZero(mature_mask3)
            immature_area3 = cv2.countNonZero(immature_mask3)
            aged_area3 = cv2.countNonZero(aged_mask3)
            total_area3 = mature_area3 + aged_area3 + immature_area3
        

            moist_mask3 = cv2.inRange(hsv3, moist_lower1, moist_upper1)
            dry_mask3 = cv2.inRange(hsv3, dry_lower1, dry_upper1)

            moist_area3 = cv2.countNonZero(moist_mask3)
            dry_area3 = cv2.countNonZero(dry_mask3)

            total_moist_area3 = moist_area3 + dry_area3

            mature_percent3 = (mature_area3 / total_area3) * 100 if total_area3 > 0 else 0
            aged_percent3= (aged_area3 / total_area3) * 100 if total_area3 > 0 else 0
            immature_percent3 = (immature_area3 / total_area3) * 100 if total_area3 > 0 else 0

            moist_percent3 = (moist_area3 / total_moist_area3) * 10 if total_moist_area3 > 0 else 0
            dry_percent3 = (dry_area3 / total_moist_area3) * 100 if total_moist_area3 > 0 else 0        



            # ------------------- Displaying Outcomes --------------------- #

            col1,col2,col3 = st.columns(3)
            with col1:
                st.image(immature_mask1, caption="Immatured Area", use_column_width=True)
            with col2:
                st.image(aged_mask,caption="Aged Region",use_column_width=True)
            with col3:
                st.image(moist_mask, caption="Moist Region",use_column_width=True)
            
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col4,col5 = st.columns(2)
            with col4:
                st.image(frame_black_dots_rgb, caption="Black Dots Highlighted", use_column_width=True)
            with col5:
                st.image(frame_faint_black_dots_rgb, caption="Faint Black Dots Highlighted", use_column_width=True)

            average_moist_percent = (moist_percent1 + moist_percent2 + moist_percent3 + moist_percent) / 4
            st.markdown(f"### Moist Percentage: {average_moist_percent:.2f}")
            average_mature_percent = (mature_percent1 + mature_percent2 + mature_percent3 + mature_percent) / 4
            st.markdown(f"### Mature percentage: {average_mature_percent:.2f}%")
            st.markdown(f"### Number of Black Dots: {len(black_dots)}")
            st.markdown(f"### Number of Faint Black Dots: {len(faint_black_dots)}")
            if mature_percent > minimum_threshold_mature_percentage and (2 < average_moist_percent < 10):
                st.success("### Final Result by : Fresh Matured Toor Daal")
            else:
                st.warning("### Final Result by : Poor Quality Toor Daal")
            print(moist_percent, "\n", moist_percent1, "\n", moist_percent2, "\n", moist_percent3)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ################## ~~~~~~~~~~~~~~~~~~~~~~~

# #######~~~~~~~~~~~~~~~~~~~~~########~~~~~~~~~~~~~~~~~



elif option == "Moong Daal":

    photo1 = st.camera_input("Take first photo of Moong daal")
    st.write("\n")
    st.write("\n")
    if photo1:
        st.write("Image 1 of Moong Daal captured successfully!")
        st.write("\n")
        photo2 = st.camera_input("Take second photo of Moong Daal.")
        st.write("\n")
        st.write("\n")
        if photo2:
            st.write("Image 2 of Moong Daal captured successfully!")
            st.write("\n")
            photo3 = st.camera_input("Take third photo of Moong Daal.")
            if photo3:
                st.write("Image 3 captured successfully!")
            st.write("\n")
            st.write("\n")
    else:
        st.write("Please take first photo of Moong Daal.")

    video_upload = st.file_uploader("Upload a video file of Moong Daal", type=["mp4", "avi", "mov"])
    st.write("\n")
    st.write("\n")

    if video_upload is not None:
        
        st.title("Moong Daal Quality Assessment")
        st.write("\n")
        st.write("\n")

        inputNumber = st.number_input("Enter your moisture content (in percentage) : ")
        st.write("\n")
        st.write("\n")
        
        if inputNumber:

            # Save the uploaded video temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(video_upload.read())
            video_path = temp_file.name  

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Error: Could not open the uploaded video.")
                st.stop()

            st.success("Moong Daal Video uploaded successfully! Processing...")

            minimum_threshold_mature_percentage = 70

            mature_lower = np.array([20, 100, 150])  # Lower HSV limit
            mature_upper = np.array([35, 255, 255])  # Upper HSV limit

            immature_lower = np.array([20, 70, 70])
            immature_upper = np.array([30, 200, 150])

            aged_lower = np.array([35, 100, 100])
            aged_upper = np.array([50, 255, 255])


            moist_lower = np.array([22, 120, 100])
            moist_upper = np.array([32, 200, 200])
            dry_lower = np.array([25, 150, 180])
            dry_upper = np.array([35, 255, 255])


            # Placeholders for dynamic updates
            frame_placeholder = st.empty()
            moist_percent_placeholder = st.empty()
            mature_percent_placeholder = st.empty()
            result_placeholder = st.empty()

            # Initialize variables for aggregation
            total_mature_area = 0
            total_immature_area = 0
            total_aged_area = 0
            total_moist_area = 0
            total_dry_area = 0
            total_area = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Process video frames
            step = 10  # Process every 10th frame
            for frame_idx in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

            # Resize frame for better performance
            frame_resized = cv2.resize(frame, (640, 360))

            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

            mature_mask = cv2.inRange(hsv, mature_lower, mature_upper)
            immature_mask = cv2.inRange(hsv, immature_lower, immature_upper)
            aged_mask = cv2.inRange(hsv, aged_lower, aged_upper)

            mature_area = cv2.countNonZero(mature_mask)
            immature_area = cv2.countNonZero(immature_mask)
            aged_area = cv2.countNonZero(aged_mask)

            frame_total_area = mature_area + immature_area + aged_area

            total_mature_area += mature_area
            total_immature_area += immature_area
            total_aged_area += aged_area
            total_area += frame_total_area


            frame_placeholder.image(frame_resized, channels="BGR", use_column_width=True)

            mature_percent = (total_mature_area / total_area) * 100 if total_area > 0 else 0
            immature_percent = (total_immature_area / total_area) * 100 if total_area > 0 else 0
            aged_percent = (total_aged_area / total_area) * 100 if total_area > 0 else 0        


            #~~~~~~~~~~~~~~~~~~~~~~


        # ~~~~~~~~~~~~~~~ Black Dot Analysis ~~~~~~~~~~~~~~~~~~~``

            blur = cv2.medianBlur(frame, 5)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            black_dots = [c for c in contours if cv2.contourArea(c) > 5]

            frame_black_dots = frame.copy()
            cv2.drawContours(frame_black_dots, black_dots, -1, (0, 255, 0), 2)

            frame_black_dots_rgb = cv2.cvtColor(frame_black_dots, cv2.COLOR_BGR2RGB)


            #~~~~~~~~~~~~~~~~~ Faint Black Dots ~~~~~~~~~~~~~~~~~

            # Faint Black Dot Analysis
            blur = cv2.medianBlur(frame, 5)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            faint_black_dots = [c for c in contours if cv2.contourArea(c) > 5]

            frame_faint_black_dots = frame.copy()
            cv2.drawContours(frame_faint_black_dots, faint_black_dots, -1, (0, 255, 0), 2)

            frame_faint_black_dots_rgb = cv2.cvtColor(frame_faint_black_dots, cv2.COLOR_BGR2RGB)



            # -------------------------- Photo 1 Analysis -------------------- #

            image1 = Image.open(photo1)
            image_np1 = np.array(image1)

            # Convert to NumPy array for OpenCV
            resized_image1 = cv2.resize(image_np1, (500, 500))

            hsv1 = cv2.cvtColor(resized_image1, cv2.COLOR_RGB2HSV)

            # mature_lower = np.array([25, 80, 80])
            # mature_upper = np.array([35, 255, 255])
            mature_lower1 = np.array([20, 70, 70])  # Lower saturation and value for darker/lighter yellows
            mature_upper1 = np.array([40, 255, 255])  # Higher hue to capture yellow-orange tones


            immature_lower1 = np.array([20, 70, 70]) 
            immature_upper1 = np.array([30, 200, 150])

            aged_lower1 = np.array([35, 100, 100])
            aged_upper1 = np.array([50, 255, 255])


            mature_mask1 = cv2.inRange(hsv1, mature_lower1, mature_upper1)
            immature_mask1 = cv2.inRange(hsv1, immature_lower1, immature_upper1)
            aged_mask1 = cv2.inRange(hsv1, aged_lower1, aged_upper1)
            

            mature_area1 = cv2.countNonZero(mature_mask1)
            immature_area1 = cv2.countNonZero(immature_mask1)
            aged_area1 = cv2.countNonZero(aged_mask1)
            total_area1 = mature_area1 + aged_area1 + immature_area1


            moist_lower1 = np.array([22, 120, 100])
            moist_upper1 = np.array([32, 200, 200])
            
            dry_lower1 = np.array([25, 150, 180])
            dry_upper1 = np.array([35, 255, 255])
        

            moist_mask1 = cv2.inRange(hsv, moist_lower, moist_upper)
            dry_mask1 = cv2.inRange(hsv, dry_lower, dry_upper)

            moist_area1 = cv2.countNonZero(moist_mask1)
            dry_area1 = cv2.countNonZero(dry_mask1)

            total_moist_area1 = moist_area1 + dry_area1

            mature_percent1 = (mature_area1 / total_area1) * 100 if total_area1 > 0 else 0
            aged_percent1 = (aged_area1 / total_area1) * 100 if total_area1 > 0 else 0
            immature_percent1 = (immature_area1 / total_area1) * 100 if total_area1 > 0 else 0

            moist_percent1 = (moist_area1 / total_moist_area1) * 10 if total_moist_area1 > 0 else 0
            dry_percent1 = (dry_area1 / total_moist_area1) * 100 if total_moist_area1 > 0 else 0        



            # ---------------------------- Photo 2 Anlysis ---------------------- #


            image2 = Image.open(photo2)
            image_np2 = np.array(image2)

            # Convert to NumPy array for OpenCV
            resized_image2 = cv2.resize(image_np2, (500, 500))

            hsv2 = cv2.cvtColor(resized_image2, cv2.COLOR_RGB2HSV)


            mature_mask2 = cv2.inRange(hsv2, mature_lower1, mature_upper1)
            immature_mask2 = cv2.inRange(hsv2, immature_lower1, immature_upper1)
            aged_mask2 = cv2.inRange(hsv2, aged_lower1, aged_upper1)
            

            mature_area2 = cv2.countNonZero(mature_mask2)
            immature_area2 = cv2.countNonZero(immature_mask2)
            aged_area2 = cv2.countNonZero(aged_mask2)
            total_area2 = mature_area2 + aged_area2 + immature_area2
        

            moist_mask2 = cv2.inRange(hsv2, moist_lower1, moist_upper1)
            dry_mask2 = cv2.inRange(hsv2, dry_lower1, dry_upper1)

            moist_area2 = cv2.countNonZero(moist_mask2)
            dry_area2 = cv2.countNonZero(dry_mask2)

            total_moist_area2 = moist_area2 + dry_area2

            mature_percent2 = (mature_area2 / total_area2) * 100 if total_area2 > 0 else 0
            aged_percent2 = (aged_area2 / total_area2) * 100 if total_area2 > 0 else 0
            immature_percent2 = (immature_area2 / total_area2) * 100 if total_area2 > 0 else 0

            moist_percent2 = (moist_area2 / total_moist_area2) * 10 if total_moist_area2 > 0 else 0
            dry_percent2 = (dry_area2 / total_moist_area2) * 100 if total_moist_area2 > 0 else 0        



            # ---------------------- Photo 3 Analysis ------------------#


            image3 = Image.open(photo3)
            image_np3 = np.array(image3)

            # Convert to NumPy array for OpenCV
            resized_image3 = cv2.resize(image_np3, (500, 500))

            hsv3 = cv2.cvtColor(resized_image3, cv2.COLOR_RGB2HSV)

            mature_mask3 = cv2.inRange(hsv3, mature_lower1, mature_upper1)
            immature_mask3 = cv2.inRange(hsv3, immature_lower1, immature_upper1)
            aged_mask3 = cv2.inRange(hsv3, aged_lower1, aged_upper1)
            

            mature_area3 = cv2.countNonZero(mature_mask3)
            immature_area3 = cv2.countNonZero(immature_mask3)
            aged_area3 = cv2.countNonZero(aged_mask3)
            total_area3 = mature_area3 + aged_area3 + immature_area3
        

            moist_mask3 = cv2.inRange(hsv3, moist_lower1, moist_upper1)
            dry_mask3 = cv2.inRange(hsv3, dry_lower1, dry_upper1)

            moist_area3 = cv2.countNonZero(moist_mask3)
            dry_area3 = cv2.countNonZero(dry_mask3)

            total_moist_area3 = moist_area3 + dry_area3

            mature_percent3 = (mature_area3 / total_area3) * 100 if total_area3 > 0 else 0
            aged_percent3= (aged_area2 / total_area3) * 100 if total_area3 > 0 else 0
            immature_percent3 = (immature_area3 / total_area3) * 100 if total_area3 > 0 else 0

            moist_percent3 = (moist_area3 / total_moist_area3) * 10 if total_moist_area3 > 0 else 0
            dry_percent3 = (dry_area3 / total_moist_area3) * 100 if total_moist_area3 > 0 else 0        



            # ------------------- Displaying Outcomes --------------------- #


            col1,col2,col3 = st.columns(3)
            with col1:
                st.image(immature_mask1, caption="Immatured Area", use_column_width=True)
            with col2:
                st.image(aged_mask,caption="Aged Region",use_column_width=True)
            with col3:
                st.image(moist_mask1, caption="Moist Region",use_column_width=True)
            
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col4,col5 = st.columns(2)
            with col4:
                st.image(frame_black_dots_rgb, caption="Black Dots Highlighted", use_column_width=True)
            with col5:
                st.image(frame_faint_black_dots_rgb, caption="Faint Black Dots Highlighted", use_column_width=True)



            average_moist_percent = (moist_percent1 + moist_percent2 + moist_percent3) / 3

            average_mature_percent = (mature_percent1 + mature_percent2 + mature_percent3 + mature_percent) / 4
            st.markdown(f"### Mature percentage: {average_mature_percent:.2f}%")
            st.markdown(f"### Moist Percentage: {average_moist_percent:.2f}")
            st.markdown(f"### Number of Black Dots: {len(black_dots)}")
            st.markdown(f"### Number of Faint Black Dots: {len(faint_black_dots)}")
            if mature_percent > minimum_threshold_mature_percentage and (2 < average_moist_percent < 10):
                st.success("### Final Result by : Fresh Matured Moong Daal")
            else:
                st.warning("### Final Result by : Poor Quality Moong Daal")
        
        



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ################## ~~~~~~~~~~~~~~~~~~~~~~~

# #######~~~~~~~~~~~~~~~~~~~~~########~~~~~~~~~~~~~~~~~




elif option == "Harbhara Daal":

    photo1 = st.camera_input("Take first photo of Harbhara daal")
    st.write("\n")
    st.write("\n")
    if photo1:
        st.write("Image 1 of Harbhara daal captured successfully!")
        st.write("\n")
        photo2 = st.camera_input("Take second photo of Harbhara daal.")
        st.write("\n")
        st.write("\n")
        if photo2:
            st.write("Image 2 of Harbhara daal captured successfully!")
            st.write("\n")
            photo3 = st.camera_input("Take third photo of Harbhara daal.")
            if photo3:
                st.write("Image 3 captured successfully!")
            st.write("\n")
            st.write("\n")
    else:
        st.write("Please take first photo of Harbhara daal.")

    video_upload = st.file_uploader("Upload a video file of Harbhara Daal", type=["mp4", "avi", "mov"])
    st.write("\n")
    st.write("\n")

    if video_upload is not None:
        
        st.title("Harbhara Daal Quality Assessment")
        st.write("\n")
        st.write("\n")

        inputNumber = st.number_input("Enter your moisture content (in percentage) : ")
        st.write("\n")
        st.write("\n")
        
        if inputNumber:

            # Save the uploaded video temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(video_upload.read())
            video_path = temp_file.name  

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Error: Could not open the uploaded video.")
                st.stop()

            st.success("Harbhara Daal Video uploaded successfully! Processing...")

            minimum_threshold_mature_percentage = 70

            mature_lower = np.array([20, 100, 120])  # Lower range
            mature_upper = np.array([40, 255, 255])  # Upper range


            immature_lower = np.array([20, 70, 70])
            immature_upper = np.array([30, 200, 150])

            aged_lower = np.array([35, 100, 100])
            aged_upper = np.array([50, 255, 255])


            moist_lower = np.array([22, 120, 100])
            moist_upper = np.array([32, 200, 200])
            dry_lower = np.array([25, 150, 180])
            dry_upper = np.array([35, 255, 255])


            # Placeholders for dynamic updates
            frame_placeholder = st.empty()
            moist_percent_placeholder = st.empty()
            mature_percent_placeholder = st.empty()
            result_placeholder = st.empty()

            # Initialize variables for aggregation
            total_mature_area = 0
            total_immature_area = 0
            total_aged_area = 0
            total_moist_area = 0
            total_dry_area = 0
            total_area = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Process video frames
            step = 10  # Process every 10th frame
            for frame_idx in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

            # Resize frame for better performance
            frame_resized = cv2.resize(frame, (640, 360))


            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

            mature_mask = cv2.inRange(hsv, mature_lower, mature_upper)
            immature_mask = cv2.inRange(hsv, immature_lower, immature_upper)
            aged_mask = cv2.inRange(hsv, aged_lower, aged_upper)

            moist_mask = cv2.inRange(hsv, moist_lower, moist_upper)
            dry_mask = cv2.inRange(hsv, dry_lower, dry_upper)

            mature_area = cv2.countNonZero(mature_mask)
            immature_area = cv2.countNonZero(immature_mask)
            aged_area = cv2.countNonZero(aged_mask)
            moist_area = cv2.countNonZero(moist_mask)
            dry_area = cv2.countNonZero(dry_mask)

            frame_total_area = mature_area + immature_area + aged_area

            total_mature_area += mature_area
            total_immature_area += immature_area
            total_aged_area += aged_area
            total_moist_area += moist_area
            total_dry_area += dry_area
            total_area += frame_total_area

            moist_percent = (moist_area / (moist_area + dry_area)) * 100 if (moist_area + dry_area) > 0 else 0
            dry_percent = (dry_area / (moist_area + dry_area)) * 100 if (moist_area + dry_area) > 0 else 0

            frame_placeholder.image(frame_resized, channels="BGR", use_column_width=True)

            mature_percent = (total_mature_area / total_area) * 100 if total_area > 0 else 0
            immature_percent = (total_immature_area / total_area) * 100 if total_area > 0 else 0
            aged_percent = (total_aged_area / total_area) * 100 if total_area > 0 else 0

            #~~~~~~~~~~~~~~~~~~~~~~


        # ~~~~~~~~~~~~~~~ Black Dot Analysis ~~~~~~~~~~~~~~~~~~~``

            blur = cv2.medianBlur(frame, 5)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            black_dots = [c for c in contours if cv2.contourArea(c) > 5]

            frame_black_dots = frame.copy()
            cv2.drawContours(frame_black_dots, black_dots, -1, (0, 255, 0), 2)

            frame_black_dots_rgb = cv2.cvtColor(frame_black_dots, cv2.COLOR_BGR2RGB)


            #~~~~~~~~~~~~~~~~~ Faint Black Dots ~~~~~~~~~~~~~~~~~

            # Faint Black Dot Analysis
            blur = cv2.medianBlur(frame, 5)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            faint_black_dots = [c for c in contours if cv2.contourArea(c) > 5]

            frame_faint_black_dots = frame.copy()
            cv2.drawContours(frame_faint_black_dots, faint_black_dots, -1, (0, 255, 0), 2)

            frame_faint_black_dots_rgb = cv2.cvtColor(frame_faint_black_dots, cv2.COLOR_BGR2RGB)



            # -------------------------- Photo 1 Analysis -------------------- #

            image1 = Image.open(photo1)
            image_np1 = np.array(image1)

            # Convert to NumPy array for OpenCV
            resized_image1 = cv2.resize(image_np1, (500, 500))

            hsv1 = cv2.cvtColor(resized_image1, cv2.COLOR_RGB2HSV)


            mature_lower1 = np.array([20, 100, 120])  # Lower range
            mature_upper1 = np.array([40, 255, 255])  # Upper range


            immature_lower1 = np.array([20, 70, 70]) 
            immature_upper1 = np.array([30, 200, 150])

            aged_lower1 = np.array([35, 100, 100])
            aged_upper1 = np.array([50, 255, 255])


            mature_mask1 = cv2.inRange(hsv1, mature_lower1, mature_upper1)
            immature_mask1 = cv2.inRange(hsv1, immature_lower1, immature_upper1)
            aged_mask1 = cv2.inRange(hsv1, aged_lower1, aged_upper1)
            

            mature_area1 = cv2.countNonZero(mature_mask1)
            immature_area1 = cv2.countNonZero(immature_mask1)
            aged_area1 = cv2.countNonZero(aged_mask1)
            total_area1 = mature_area1 + aged_area1 + immature_area1


            moist_lower1 = np.array([22, 120, 100])
            moist_upper1 = np.array([32, 200, 200])
            
            dry_lower1 = np.array([25, 150, 180])
            dry_upper1 = np.array([35, 255, 255])
        

            moist_mask1 = cv2.inRange(hsv, moist_lower, moist_upper)
            dry_mask1 = cv2.inRange(hsv, dry_lower, dry_upper)

            moist_area1 = cv2.countNonZero(moist_mask1)
            dry_area1 = cv2.countNonZero(dry_mask1)

            total_moist_area1 = moist_area1 + dry_area1

            mature_percent1 = (mature_area1 / total_area1) * 100 if total_area1 > 0 else 0
            aged_percent1 = (aged_area1 / total_area1) * 100 if total_area1 > 0 else 0
            immature_percent1 = (immature_area1 / total_area1) * 100 if total_area1 > 0 else 0

            moist_percent1 = (moist_area1 / total_moist_area1) * 10 if total_moist_area1 > 0 else 0
            dry_percent1 = (dry_area1 / total_moist_area1) * 100 if total_moist_area1 > 0 else 0        



            # ---------------------------- Photo 2 Anlysis ---------------------- #


            image2 = Image.open(photo2)
            image_np2 = np.array(image2)

            # Convert to NumPy array for OpenCV
            resized_image2 = cv2.resize(image_np2, (500, 500))

            hsv2 = cv2.cvtColor(resized_image2, cv2.COLOR_RGB2HSV)


            mature_mask2 = cv2.inRange(hsv2, mature_lower1, mature_upper1)
            immature_mask2 = cv2.inRange(hsv2, immature_lower1, immature_upper1)
            aged_mask2 = cv2.inRange(hsv2, aged_lower1, aged_upper1)
            

            mature_area2 = cv2.countNonZero(mature_mask2)
            immature_area2 = cv2.countNonZero(immature_mask2)
            aged_area2 = cv2.countNonZero(aged_mask2)
            total_area2 = mature_area2 + aged_area2 + immature_area2
        

            moist_mask2 = cv2.inRange(hsv2, moist_lower1, moist_upper1)
            dry_mask2 = cv2.inRange(hsv2, dry_lower1, dry_upper1)

            moist_area2 = cv2.countNonZero(moist_mask2)
            dry_area2 = cv2.countNonZero(dry_mask2)

            total_moist_area2 = moist_area2 + dry_area2

            mature_percent2 = (mature_area2 / total_area2) * 100 if total_area2 > 0 else 0
            aged_percent2 = (aged_area2 / total_area2) * 100 if total_area2 > 0 else 0
            immature_percent2 = (immature_area2 / total_area2) * 100 if total_area2 > 0 else 0

            moist_percent2 = (moist_area2 / total_moist_area2) * 10 if total_moist_area2 > 0 else 0
            dry_percent2 = (dry_area2 / total_moist_area2) * 100 if total_moist_area2 > 0 else 0        



            # ---------------------- Photo 3 Analysis ------------------#


            image3 = Image.open(photo3)
            image_np3 = np.array(image3)

            # Convert to NumPy array for OpenCV
            resized_image3 = cv2.resize(image_np3, (500, 500))

            hsv3 = cv2.cvtColor(resized_image3, cv2.COLOR_RGB2HSV)

            mature_mask3 = cv2.inRange(hsv3, mature_lower1, mature_upper1)
            immature_mask3 = cv2.inRange(hsv3, immature_lower1, immature_upper1)
            aged_mask3 = cv2.inRange(hsv3, aged_lower1, aged_upper1)
            

            mature_area3 = cv2.countNonZero(mature_mask3)
            immature_area3 = cv2.countNonZero(immature_mask3)
            aged_area3 = cv2.countNonZero(aged_mask3)
            total_area3 = mature_area3 + aged_area3 + immature_area3
        

            moist_mask3 = cv2.inRange(hsv3, moist_lower1, moist_upper1)
            dry_mask3 = cv2.inRange(hsv3, dry_lower1, dry_upper1)

            moist_area3 = cv2.countNonZero(moist_mask3)
            dry_area3 = cv2.countNonZero(dry_mask3)

            total_moist_area3 = moist_area3 + dry_area3

            mature_percent3 = (mature_area3 / total_area3) * 100 if total_area3 > 0 else 0
            aged_percent3= (aged_area3 / total_area3) * 100 if total_area3 > 0 else 0
            immature_percent3 = (immature_area3 / total_area3) * 100 if total_area3 > 0 else 0

            moist_percent3 = (moist_area3 / total_moist_area3) * 10 if total_moist_area3 > 0 else 0
            dry_percent3 = (dry_area3 / total_moist_area3) * 100 if total_moist_area3 > 0 else 0        



            # ------------------- Displaying Outcomes --------------------- #


            col1,col2,col3 = st.columns(3)
            with col1:
                st.image(immature_mask1, caption="Immatured Area", use_column_width=True)
            with col2:
                st.image(aged_mask,caption="Aged Region",use_column_width=True)
            with col3:
                st.image(moist_mask1, caption="Moist Region",use_column_width=True)
            
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col4,col5 = st.columns(2)
            with col4:
                st.image(frame_black_dots_rgb, caption="Black Dots Highlighted", use_column_width=True)
            with col5:
                st.image(frame_faint_black_dots_rgb, caption="Faint Black Dots Highlighted", use_column_width=True)
        

            average_moist_percent = (moist_percent1 + moist_percent2 + moist_percent3) / 3
            st.markdown(f"### Moist Percentage: {average_moist_percent:.2f}")
            average_mature_percent = (mature_percent1 + mature_percent2 + mature_percent3 + mature_percent) / 4
            st.markdown(f"### Mature percentage: {mature_percent:.2f}%")
            st.markdown(f"### Number of Black Dots: {len(black_dots)}")
            st.markdown(f"### Number of Faint Black Dots: {len(faint_black_dots)}")
            if mature_percent > minimum_threshold_mature_percentage and (2 < average_moist_percent < 10):
                st.success("### Final Result by : Fresh Matured Harbhara Daal")
            else:
                st.warning("### Final Result by : Poor Quality Harbhara Daal")

