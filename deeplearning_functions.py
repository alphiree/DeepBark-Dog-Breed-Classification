from deeplearning_packages import *

def multiclass_predict_from_image_with_comparison(model,image,image_size,class_names,comparison_path,picture_title):
    plot_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plot_image_2 = cv2.resize(plot_image,(400,400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Preprocessing
    image = tf.image.resize(image,image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    ## Predicting model
    score = model.predict(image_array)
    # print(f'The score list is {score}')
    index = np.argmax(score)
    score = np.ravel(score)
    rounded_score = np.round(score[index],4)
    print(f'rounded off score is: {rounded_score}')

    breed = class_names[index]

    comparison_folder = os.path.join(comparison_path, class_names[index])

    print('File Count inside the comparison path:')
    print(len([entry for entry in os.listdir(comparison_folder) if os.path.isfile(os.path.join(comparison_folder, entry))]))

    image_list = []

    for images in os.listdir(comparison_folder):
        image_list.append(images)

    # read all the images
    # we are going to take 4 images only
    image1=cv2.imread(os.path.join(comparison_folder, image_list[0]))
    image2=cv2.imread(os.path.join(comparison_folder, image_list[1]))
    image3=cv2.imread(os.path.join(comparison_folder, image_list[2]))
    image4=cv2.imread(os.path.join(comparison_folder, image_list[3]))

    # make all the images of same size 
    #so we will use resize function
    image1=cv2.resize(image1,(200,200))
    image2=cv2.resize(image2,(200,200))
    image3=cv2.resize(image3,(200,200))
    image4=cv2.resize(image4,(200,200))

    Horizontal1=np.hstack([image1,image2])
    Horizontal2=np.hstack([image3,image4])

    Vertical_attachment=np.vstack([Horizontal1,Horizontal2])

    final_image = cv2.cvtColor(Vertical_attachment,cv2.COLOR_BGR2RGB)

    ## Plotting it
    fig, axes = plt.subplots(1, ## Number of columns
                            2, ## Number of rows
                            figsize=(12,6), ## Figure Size
                            )

    ## Plotting the Subplots
    axes[0].imshow(plot_image_2)
    axes[0].title.set_text(f'Predicted Class: {breed}')
    axes[0].axis("off")
    
    axes[1].imshow(final_image)
    axes[1].title.set_text(breed)
    axes[1].axis("off")

    ## If you have more than 1 columns and want to put title in those specific columns
    plt.suptitle(picture_title)

    plt.show()

def model_train_evaluate(model,
                    train_ds,
                    val_ds,
                    test_ds,
                    epochs,
                    model_name,
                    save_model = True,
                    plot_train = True,
                    save_plot_train = True,
                    plot_confusion_matrix = True,
                    save_confusion_matrix = True,
                        ):
    start_time = datetime.now()
    ## We will set our epochs to be 10 epochs
    history = model.fit(train_ds,
                        epochs = 10,
                        validation_data = val_ds,
                        # callbacks = callback_list,
                    )

    end_time = datetime.now()


    print('')
    print('Duration: {}'.format(end_time - start_time))

    if save_model:
        ## Saving the model
        model.save(f'saved-models\\{model_name}.h5')
    
    if plot_train:
        ## Plotting the accuracy and loss
        plot_acc_and_loss(history)

        ## If you want to set a title
        plt.suptitle(model_name)
    
        if save_plot_train:
            save_fig(
            #folder name, if none put '' instead
            folder = fig_folder, 
            name = f'accuracy_loss_{model_name}',
            format = 'pdf',
            dpi = 1000,
            )
        else:
            plt.show()
    
    ## Creating the time it takes to run the model
    duration = end_time - start_time
    duration = str(duration).split('.', 2)[0]

    ## Predicting and Evaluating the model
    train_loss, train_accuracy = model.evaluate(train_ds)
    loss, accuracy = model.evaluate(test_ds)
    test_score = model.predict(test_ds)


    ## Saving the model result into dictionary
    model_result = {
            'Training Time':duration,
            'Train Set Accuracy':train_accuracy,
            'Test Set Accuracy':accuracy,
                }

    test_result = []
    y_test_result = []

    for i in range(len(test_score)):
        index = np.argmax(test_score[i])
        test_result.append(index)

    for i in range(len(y_test)):
        index = np.argmax(y_test[i])
        y_test_result.append(index)

    result_df = pd.DataFrame({'Predicted Result':test_result,
    'Actual Value': y_test_result,}
                )

    report = classification_report(
        y_true = y_test_result,
        y_pred = test_result,
        output_dict = True,
    )
    print(classification_report(
        y_true = y_test_result,
        y_pred = test_result,
    ))

    report_df = pd.DataFrame(report).T.round(3)

    ## convert it into latex
    print(report_df.to_latex(
                                index = True,
                                multicolumn = True,
                                caption = f'MobileNetV2 Classification Report in {dataset_name}',
                                label = f'tab:{model_name}_{dataset_name}',
                                escape =  True,
                                column_format = 'XXXXX',
    ))

    if plot_confusion_matrix:
        cm = confusion_matrix(
        y_true = y_test_result,
        y_pred = test_result,
        )

        fig,ax = plt.subplots(figsize=(9,7)) 

        conf = sns.heatmap(cm, 
                        annot=True,
                        fmt='1',
                        #    cmap='summer',
                        cmap=sns.cubehelix_palette(rot=-.2,as_cmap=True),
                        linewidths=1,
                        linecolor='#0B021E',
                        )
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title(f'Confusion Matrix using {model_name}')

        if save_confusion_matrix:
            save_fig(
                    #folder name, if none put '' instead
                    folder = fig_folder, 
                    name = f'conf_matrix_{model_name}',
                    format = 'pdf',
                    dpi = 1000,
                    )
        
        else:
            plt.show()
    
    return model_result,result_df


def model_evaluate(model,
                    train_ds,
                    val_ds,
                    test_ds,
                    model_name,
                    plot_confusion_matrix = True,
                    save_confusion_matrix = False,
                        ):
    ## Predicting and Evaluating the model
    loss, accuracy = model.evaluate(test_ds)
    test_score = model.predict(test_ds)

    test_result = []
    y_test_result = []

    for i in range(len(test_score)):
        index = np.argmax(test_score[i])
        test_result.append(index)

    for i in range(len(y_test)):
        index = np.argmax(y_test[i])
        y_test_result.append(index)

    result_df = pd.DataFrame({'Predicted Result':test_result,
    'Actual Value': y_test_result,}
                )

    report = classification_report(
        y_true = y_test_result,
        y_pred = test_result,
        output_dict = True,
    )
    print(classification_report(
        y_true = y_test_result,
        y_pred = test_result,
    ))

    report_df = pd.DataFrame(report).T.round(3)

    ## convert it into latex
    print(report_df.to_latex(
                                index = True,
                                multicolumn = True,
                                caption = f'MobileNetV2 Classification Report in {dataset_name}',
                                label = f'tab:{model_name}_{dataset_name}',
                                escape =  True,
                                column_format = 'XXXXX',
    ))

    if plot_confusion_matrix:
        cm = confusion_matrix(
        y_true = y_test_result,
        y_pred = test_result,
        )

        fig,ax = plt.subplots(figsize=(9,7)) 

        conf = sns.heatmap(cm, 
                        annot=True,
                        fmt='1',
                        #    cmap='summer',
                        cmap=sns.cubehelix_palette(rot=-.2,as_cmap=True),
                        linewidths=1,
                        linecolor='#0B021E',
                        )
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title(f'Confusion Matrix using {model_name}')

        if save_confusion_matrix:
            save_fig(
                    #folder name, if none put '' instead
                    folder = fig_folder, 
                    name = f'conf_matrix_{model_name}',
                    format = 'pdf',
                    dpi = 1000,
                    )
        
        else:
            plt.show()
    
    return result_df
    

## Defining Functions

## For Model Training and evaluating


## counting images per class
def count_images_per_class(train_dir, val_dir, test_dir):
    overall_counts = {}
    for directory in [train_dir, val_dir, test_dir]:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    class_name = os.path.basename(root)
                    if class_name in overall_counts:
                        overall_counts[class_name] += 1
                    else:
                        overall_counts[class_name] = 1

    print("Overall Counts:")
    for class_name, count in overall_counts.items():
        print(f"{class_name}: {count}")


## plotting accuracy and loss
def plot_acc_and_loss(fitted_model):
    train_accuracy = fitted_model.history['accuracy']
    val_accuracy = fitted_model.history['val_accuracy']

    train_loss = fitted_model.history['loss']
    val_loss = fitted_model.history['val_loss']

    length = len(train_accuracy)

    fig, axes = plt.subplots(1,2, figsize=(12,6))

    axes[0].plot(np.linspace(start = 1, stop = length, num = length), train_accuracy, label='Train set',marker='.')
    axes[0].plot(np.linspace(start = 1, stop = length, num = length), val_accuracy, label='Validation set',marker='.')
    axes[0].set(xlabel = 'Epoch',ylabel = 'Accuracy',xticks = np.linspace(start = 1, stop = length, num = length))
    axes[0].title.set_text('Training and Validation Accuracy')
    axes[0].legend()

    axes[1].plot(np.linspace(start = 1, stop = length, num = length), train_loss, label='Train loss',marker='.')
    axes[1].plot(np.linspace(start = 1, stop = length, num = length), val_loss, label='Validation loss',marker='.')
    axes[1].set(xlabel = 'Epoch',ylabel = 'Loss',xticks = np.linspace(start = 1, stop = length, num = length))
    axes[1].title.set_text('Training and Validation Loss')
    axes[1].legend()

    
## read image
def read_image(path):
    image = cv2.imread(path)
    return image

## read images from a folder
def read_images_from_folder(folder_path,
                            img_format,
                            ):
    folder_path = folder_path
    img_format = img_format
    list_images = os.listdir(folder_path)
    num_files = len(list_images)


    print(f'There are {num_files} files in this folder')
    print('')

    image_num = 0 
    for images in list_images:
        # check if the image ends with png
        if (images.endswith(img_format)):
            image_num += 1

    print(f'There are {image_num} images ending in {img_format} in this folder')
    print('')

    print('These are some of the image names in the folder')
    print(list_images[:5])

    return list_images

## Function to show images
def show(file,name='image'):
    print(file.shape)
    cv2.imshow(name, file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## Show images inline
def show_inline(image, 
                rgb = True,
                resized = False):
    print(image.shape)
    print(f'lowest value of pixel is: {image.min()}')
    print(f'highest value of pixel is: {image.max()}')

    if rgb == False:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    if resized:
        plt.imshow(image)

    else:
        plt.imshow(image.astype('int'))

    plt.show()

## get train test from the batch dataset
def train_test(dataset):
    X = np.concatenate([x for x, y in dataset], axis=0)
    y = np.concatenate([y for x, y in dataset], axis=0)

    return X,y

## Plot some images in a batch function.
def plot_batch(batch,labels,class_names,resized = False):
    fig, axes = plt.subplots(1,4, figsize=(18,4))

    
    if resized:
        for index, image in enumerate(batch[:4]):
            max_index = np.argmax(labels[index])
            class_type = class_names[max_index]
            axes[index].imshow(image)
            axes[index].title.set_text(class_type)

    else:
        for index, image in enumerate(batch[:4]):
            max_index = np.argmax(labels[index])
            class_type = class_names[max_index]
            axes[index].imshow(image.astype('int'))
            axes[index].title.set_text(class_type)
    plt.show()

def binary_predict_from_image( model,
                   image,
                   image_size,
                   class_names,
                   actual_value = None,
                ):

    plot_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Preprocessing
    image = tf.image.resize(image,image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    ## Predicting model
    score = model.predict(image_array)
    print(f'The score list is {score}')
    score = np.ravel(score)
    rounded_score = np.round(score,4)
    print(f'rounded off score is: {rounded_score}')

    value = int(np.floor(score + 0.5))


    if actual_value:
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        axes[0].imshow(plot_image)
        axes[0].title.set_text(f'Actual Class: {actual_value}')

        axes[1].imshow(plot_image)
        axes[1].title.set_text(f'Predicted Class: {class_names[value]}')

        plt.show()
    
    else:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(plot_image)
        plt.title(f'Predicted Class: {class_names[value]}')
        plt.show()

def binary_predict_from_path( model,
                   path,
                   image_size,
                   class_names,
                   actual_value = None,
                ):

    image = cv2.imread(path)
    plot_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Preprocessing
    image = tf.image.resize(image,image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    ## Predicting model
    score = model.predict(image_array)
    print(f'The score list is {score}')
    score = np.ravel(score)
    rounded_score = np.round(score,4)
    print(f'rounded off score is: {rounded_score}')

    value = int(np.floor(score + 0.5))


    if actual_value:
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        axes[0].imshow(plot_image)
        axes[0].title.set_text(f'Actual Class: {actual_value}')

        axes[1].imshow(plot_image)
        axes[1].title.set_text(f'Predicted Class: {class_names[value]}')

        plt.show()
    
    else:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(plot_image)
        plt.title(f'Predicted Class: {class_names[value]}')
        plt.show()
    
def multiclass_predict_from_image( model,
                   image,
                   image_size,
                   class_names,
                   actual_value = None,
                ):
    plot_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Preprocessing
    image = tf.image.resize(image,image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    ## Predicting model
    score = model.predict(image_array)
    # print(f'The score list is {score}')
    index = np.argmax(score)
    score = np.ravel(score)
    rounded_score = np.round(score[index],4)
    print(f'rounded off score is: {rounded_score}')

    breed = class_names[index]

    fig = plt.figure(figsize=(6,6))
    plt.imshow(plot_image)

    if actual_value:
        plt.title(f'Predicted Class: {class_names[index]} \n Actual Class: {actual_value}')


    else:
        plt.title(f'Predicted Class: {breed}')


    plt.show()

def multiclass_predict_from_path( model,
                   path,
                   image_size,
                   class_names,
                   actual_value = None,
                ):

    image = cv2.imread(path)

    plot_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Preprocessing
    image = tf.image.resize(image,image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    ## Predicting model
    score = model.predict(image_array)
    # print(f'The score list is {score}')
    index = np.argmax(score)
    score = np.ravel(score)
    rounded_score = np.round(score[index],4)
    print(f'rounded off score is: {rounded_score}')

    breed = class_names[index]

    if actual_value:
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        axes[0].imshow(plot_image)
        axes[0].title.set_text(f'Actual Class: {actual_value}')

        axes[1].imshow(plot_image)
        axes[1].title.set_text(f'Predicted Class: {breed}')

        plt.show()
    
    else:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(plot_image)
        plt.title(f'Predicted Class: {breed}')
        plt.show()

def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def delete_folder(name):
    shutil.rmtree(name)

def delete_images_from_folder(name,folder):
    shutil.rmtree(os.path.join(folder,name))

def detect_face_from_image(image):
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_2 = image.copy()

    faces = face.detectMultiScale(gray,
                                scaleFactor = 1.3,
                                minNeighbors = 5,
                                #   minSize = (30,30)
                                )
    list_of_faces = []
    
    for (x,y,w,h) in faces:

        face_region = image_2[y:y+h, x:x+w]
        list_of_faces.append(face_region)

        cv2.rectangle(img = image,
                    pt1 = (x,y),
                    pt2 = (x+w,y+h),
                    color = (0,0,255),
                    thickness = 2,
                    )
                
    num_of_faces = len(list_of_faces)
    print(f'There are {num_of_faces} face/s in this picture')

    show_inline(image, rgb = False)

    return list_of_faces

def detect_face_from_path(path):

    image = cv2.imread(path)
    image_2 = cv2.imread(path)
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,
                                scaleFactor = 1.3,
                                minNeighbors = 5,
                                #   minSize = (30,30)
                                )

    list_of_faces = []
    
    for (x,y,w,h) in faces:

        face_region = image_2[y:y+h, x:x+w]
        list_of_faces.append(face_region)

        cv2.rectangle(img = image,
                    pt1 = (x,y),
                    pt2 = (x+w,y+h),
                    color = (0,0,255),
                    thickness = 2,
                    )
                
    num_of_faces = len(list_of_faces)
    print(f'There are {num_of_faces} face/s in this picture')

    show_inline(image, rgb = False)

    return list_of_faces
    

def generate_faces_from_webcam(name,num_images,input_folder):
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ## Initializing the video capture device. This will not turn on the camera on the device yet
    cap = cv2.VideoCapture(0)

    # Set the name and number of images to save
    name = name
    num_images = num_images
    # Create the output directories if they don't exist
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(f'{input_folder}/{name}'):
        os.makedirs(f'{input_folder}/{name}')


    # Initialize an image counter
    img_num = 0

    while True:
        ## Reading the cap. ret will return true if the device has camera. frame will return the frame by frame of the device
        ret, frame = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray,
                                    scaleFactor = 1.3,
                                    minNeighbors = 5,
                                    #   minSize = (30,30)
                                    )
        for (x,y,w,h) in faces:
            # cv2.rectangle(img = frame,
            #             pt1 = (x,y),
            #             pt2 = (x+w,y+h),
            #             color = (0,0,255),
            #             thickness = 2,
            #             )

            # Crop the face region from the frame
            face_region = frame[y:y+h, x:x+w]

            # Save the face as a separate image file
            if img_num < num_images:
                face_path = f'{input_folder}/{name}/{name}_{img_num}.jpg'
                cv2.imwrite(face_path, face_region)
                print(f'Saved {face_path}')
                img_num += 1
            
            cv2.rectangle(img = frame,
                        pt1 = (x,y),
                        pt2 = (x+w,y+h),
                        color = (0,0,255),
                        thickness = 2,
                        )

        # Display the frame with the rectangles around the detected faces
        cv2.imshow('Face Detection', frame)

        # Exit the program if the 'q' key is pressed or if the desired number of images has been saved
        if cv2.waitKey(1) == ord('q') or img_num == num_images:
            break

    # Release the VideoCapture object and close the window
    # cap.release() will end the ret and frame in cap.read()
    cap.release()
    cv2.destroyAllWindows()

def save_faces_from_folder(folder_path,
                           input_folder,
                           name,
                           img_format,
    ):
    folder_path = folder_path
    list_images = os.listdir(folder_path)

    if not os.path.exists(f'{input_folder}/{name}'):
        os.makedirs(f'{input_folder}/{name}')
    
    if not os.path.exists(f'{input_folder}/{name}_full'):
        os.makedirs(f'{input_folder}/{name}_full')

    ## Face Detection Tool
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize an image counter
    img_num = 0

    for images in list_images:
    # check if the image ends with png
        if (images.endswith(img_format)):
            single_image = os.path.join(folder_path,images)

            single_image = read_image(single_image)

            ## Converting it into grayscale
            gray = cv2.cvtColor(single_image, cv2.COLOR_BGR2GRAY)

            faces = face.detectMultiScale(gray,
                                    scaleFactor = 1.3,
                                    minNeighbors = 5,
                                    #   minSize = (30,30)
                                    )
            for (x,y,w,h) in faces:
                face_region = single_image[y:y+h, x:x+w]
                saved_path = f'{input_folder}/{name}/{name}_{img_num}.jpg'
                cv2.imwrite(saved_path, face_region)
                print(f'Saved {saved_path}')
                
                cv2.rectangle(img = single_image,
                      pt1 = (x,y),
                      pt2 = (x+w,y+h),
                      color = (0,0,255),
                      thickness = 2,
                      )
                saved_path_full = f'{input_folder}/{name}_full/{name}_{img_num}.jpg'
                cv2.imwrite(saved_path_full, single_image)

                img_num += 1

def face_detection():
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ## Initializing the video capture device. This will not turn on the camera on the device yet
    cap = cv2.VideoCapture(0)

    while True:
        ## Reading the cap. ret will return true if the device has camera. frame will return the frame by frame of the device
        ret, frame = cap.read()

        frame_2 = frame.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray,
                                    scaleFactor = 1.3,
                                    minNeighbors = 5,
                                    #   minSize = (30,30)
                                    )

        list_of_faces = []

        for (x,y,w,h) in faces:
            # Crop the face region from the frame
            face_region = frame_2[y:y+h, x:x+w]
            list_of_faces.append(face_region)

            cv2.rectangle(img = frame,
                        pt1 = (x,y),
                        pt2 = (x+w,y+h),
                        color = (0,0,255),
                        thickness = 2,
                        )

        num_of_faces = len(list_of_faces)
        num_face_text = f'{num_of_faces} faces detected'
        cv2.putText(img = frame, 
                    text = num_face_text, 
                    org=(10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.8, 
                    color=(255, 0, 0), 
                    thickness=2)

        cv2.imshow('capture',frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release the VideoCapture object and close the window
    # cap.release() will end the ret and frame in cap.read()
    cap.release()
    cv2.destroyAllWindows()

    return list_of_faces, frame_2


def webcam_capture():
    ## Initializing the video capture device. This will not turn on the camera on the device yet
    cap = cv2.VideoCapture(0)

    # Continuously process frames from the camera
    while True:
        ## Reading the cap. ret will return true if the device has camera. frame will return the frame by frame of the device
        ret, frame = cap.read()

        cv2.imshow('capture',frame)

        if cv2.waitKey(1) == ord('q'):
            ## cap.release() will end the ret and frame in cap.read()
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame

def webcam_capture_from_ip(url):
    # Continuously process frames from the camera
    while True:
        face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = cv2.resize(img,(900,600))

        cv2.imshow('capture',img)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    return img

   
def model_predict_from_webcam(model,
                              image_size,
                              class_names,
                            ):
    class_names = class_names

    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ## Initializing the video capture device. This will not turn on the camera on the device yet
    cap = cv2.VideoCapture(0)

    while True:
        ## Reading the cap. ret will return true if the device has camera. frame will return the frame by frame of the device
        ret, frame = cap.read()


        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray,
                                    scaleFactor = 1.3,
                                    minNeighbors = 5,
                                    #   minSize = (30,30)
                                    )
        list_of_faces = []

        for (x,y,w,h) in faces:
            cv2.rectangle(img = frame,
                        pt1 = (x,y),
                        pt2 = (x+w,y+h),
                        color = (0,255,0),
                        thickness = 2,
                        )
            # roi_gray = gray[y:y+h,x:x+w]

            # Crop the face region from the frame
            face_region = frame[y:y+h, x:x+w]

            list_of_faces.append(face_region)

            bgr_to_rgb = cv2.cvtColor(face_region,cv2.COLOR_BGR2RGB)
            ## Preprocessing
            image = tf.image.resize(bgr_to_rgb,image_size)
            image_array = img_to_array(image)
            image_array = tf.expand_dims(image_array, axis=0)

            ## Predicting model
            score = model.predict(image_array)
            score = np.ravel(score)
            print(f'The score list is {score}')
            index = np.argmax(score)
            face_class = class_names[index]
            score_value = score[index]


            if score[index] < 0.5:
                cv2.putText(img=frame, 
                            text='Unknown', 
                            org=(x,y-10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.8, 
                            color=(0, 255, 0), 
                            thickness=2)

            # Put the name above the rectangle in the face region
            else:
                text = f'{face_class} Acc: {np.round(np.float64(score_value),2)}'

                cv2.putText(img=frame, 
                            text = text, 
                            org=(x,y-10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.8, 
                            color=(0, 255, 0), 
                            thickness=2)

        num_of_faces = len(list_of_faces)
        num_face_text = f'{num_of_faces} faces detected'
        cv2.putText(img = frame, 
                    text = num_face_text, 
                    org=(10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.8, 
                    color=(255, 0, 0), 
                    thickness=2)

        cv2.imshow('capture',frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release the VideoCapture object and close the window
    # cap.release() will end the ret and frame in cap.read()
    cap.release()
    cv2.destroyAllWindows()

## print text to see the font
def print_text(text):
    fig, ax = plt.subplots(figsize=(6, 1), facecolor="#eefade")
    ax.text(0.5, 0.5, text, ha='center', va='center', size=40)
    ax.axis("off")
    plt.show()


## Saving figures
def save_fig(folder,name,format,dpi):
    full_path = os.path.join(folder,name)
    plt.savefig(f'{full_path}.{format}',dpi = dpi)