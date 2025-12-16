import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from vit import ViT
from config import config
import os

st.set_page_config(page_title="ViT Classificator", page_icon="🖼️", layout="wide")


# Загрузка модели
@st.cache_resource
def load_model():
    checkpoint = torch.load("model/model.ckpt", map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[6:]
        elif key.startswith("net."):
            key = key[4:]
        new_state_dict[key] = value

    model = ViT(config)
    model.load_state_dict(new_state_dict)

    model.eval()
    return model


# Загрузка меток классов
def load_class_labels():
    with open("model/class_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


# Предобработка изображения
def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


# Предсказание
def predict(image, model, labels):
    try:
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)

        return id2word[labels[top_class.item()]], top_prob.item()
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        return None, None

def get_tiny_imagenet_classes(root: str):
    labels_file = os.path.join(root, "wnids.txt")
    with open(labels_file, 'r') as f:
        classes = sorted([line.strip() for line in f])
        print((classes))
    return classes

root = "model"
classes = get_tiny_imagenet_classes(root)
id2word = {}
with open(f'{root}/words.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip('\n').split('	') for line in lines]
    for id, word in lines:
        id2word[id] = word

# Интерфейс Streamlit
def main():
    
    st.title("Классификация изображений с помощью ViT")
    st.write("Загрузите изображение для классификации")

    # Загрузка модели и меток
    with st.spinner("Загрузка модели..."):
        model = load_model()
        labels = load_class_labels()

    labels = classes

    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png", "bmp"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file is not None:
            # Отображение изображения
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_container_width=True)

            # Кнопка для предсказания
            if st.button("Классифицировать", type="primary"):
                with st.spinner("Анализ изображения..."):
                    predicted_class, confidence = predict(image, model, labels)

                    if predicted_class:
                        st.success(f"**Result:** {predicted_class}")
                        st.info(f"**Confidence:** {confidence:.2%}")

                        # Топ-3 предсказания
                        with torch.no_grad():
                            input_tensor = preprocess_image(image)
                            outputs = model(input_tensor)
                            outputs = outputs[0]
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            top_probs, top_indices = torch.topk(probabilities, 3)

                            st.write("**Топ-3 предсказания:**")
                            for i in range(3):
                                st.write(
                                    f"{i+1}. {id2word[labels[top_indices[0][i]]]}: {top_probs[0][i]:.2%}"
                                )

    with col2:
        # Информационная панель
        st.markdown("### Информация")
        st.markdown(f"**Количество классов:** {len(labels)}")
        st.markdown("**Поддерживаемые форматы:** JPG, PNG, JPEG, BMP")
        st.markdown("**Модель:** ViT")

        # Метки классов
        with st.expander("Просмотреть все классы"):
            for i in range(len(classes)):
                st.write(f"{i}: {id2word[classes[i]]}")


if __name__ == "__main__":
    main()
