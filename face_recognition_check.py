import face_recognition

# Load images from folder
image1 = face_recognition.load_image_file("data_images/image1.png")
image2 = face_recognition.load_image_file("data_images/image2.png")
image3 = face_recognition.load_image_file("data_images/image3.png")

# Encode faces
encodings1 = face_recognition.face_encodings(image1)
encodings2 = face_recognition.face_encodings(image2)
encodings3 = face_recognition.face_encodings(image3)

# Check for faces
if not encodings1 or not encodings2 or not encodings3:
    print("Could not detect a face in one or more images.")
    exit()

face1 = encodings1[0]
face2 = encodings2[0]
face3 = encodings3[0]

# Use face_recognition's built-in face_distance
distance_1_2 = face_recognition.face_distance([face2], face1)[0]
distance_1_3 = face_recognition.face_distance([face3], face1)[0]

print(f"Distance between image1 and image2: {distance_1_2:.4f}")
print(f"Distance between image1 and image3: {distance_1_3:.4f}")

# Print which is more similar
if distance_1_2 < distance_1_3:
    print("image1 is more similar to image2")
else:
    print("image1 is more similar to image3")
