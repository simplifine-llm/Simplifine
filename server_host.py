from flask import Flask, request, jsonify
from pipeline import *

# init the pipeline
path = '/Users/alikavoosi/Desktop/DEMO/newpdf.pdf'
pdb = pincecone_db_man()

app = Flask(__name__)

@app.route("/")
def hello():
    return "This is api service for Simplifine! :)"



# function to add embedding for data
@app.route("/embed", methods=['POST'])
def embed():
    data = request.get_json()
    if not data or 'data_list' not in data:
        return jsonify({"error": "No data list provided"}), 400
    
    data_list = data['data_list']
    if not isinstance(data_list, list):
        return jsonify({"error": "Provided data is not a list"}), 400
    
    response = pdb.add_data(data_list)
    return jsonify({"response": 'data succesfully added'}), 200

@app.route("/get_detail", methods=['POST'])
def get_detail_dic():
    print('inboke')
    try:
        detail = pdb.index.describe_index_stats()

        # Extract necessary attributes
        dimension = detail.dimension
        index_fullness = detail.index_fullness
        total_vector_count = detail.total_vector_count
        
        # Convert namespaces
        namespaces = {namespace_name: {"vector_count": ns.vector_count} for namespace_name, ns in detail.namespaces.items()}
        
        # Construct the dictionary
        result_dict = {
            "dimension": dimension,
            "index_fullness": index_fullness,
            "namespaces": namespaces,
            "total_vector_count": total_vector_count
        }

        print(f'got the details: {type(result_dict)}')
        return jsonify(dict(result_dict)), 200
    except:
        return jsonify({"error": "Failed to get index details"}), 400


@app.route("/ask", methods=['POST'])
def ask_query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No question provided"}), 400
    query = data['query']
    try:
        detail = pdb.search(query)
        inds = []
        for i in detail['matches']:
            inds.append(i['id'])
        # print(jsonify(detail))
        return jsonify(inds), 200
    except:
        return jsonify({"error": "Failed to get index details"}), 400
    




# @app.route("/qa")
# def qa():
#     return pdb.ask_question('what is the main idea behind the article?')

if __name__ == "__main__":
    app.run(debug=True)