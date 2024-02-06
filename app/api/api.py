def get_relevant_images(request, search_input):
    embedding_as_np = request.app.state.clip.get_text_embedding(search_input.input_text)
    to_be_sent_back = []
    response = request.app.state.qdrant.qdrant_client.search(
            collection_name="images_search",
            query_vector=embedding_as_np.tolist(),
        )
    for point in response:
        if point.score > 0.25:
            to_be_sent_back.append({"score": point.score,"image": point.payload["uri"]})
    return to_be_sent_back
