# Cleaning function for lyrics
def limpieza(x):
    import re
    
    x=x.strip('[')
    x=x.strip(']')
    x=x.strip('"')
    x=x.strip("'")
    
    x_clean = re.sub(r'\\u205f',' ', x)
    
    pattern1 = r'\[.*?\]'
    pattern2 = r'(?<=[a-z])(?=[A-Z])'
    pattern3 = r'(?<=[^\s\w])(?=\w*[A-Z])'
    
    x_clean = re.sub(pattern1, '', x_clean)
    x_clean = re.sub(pattern2, ' ', x_clean)
    x_clean = re.sub(pattern3, ' ', x_clean)
    x_clean = re.sub(r'\\', '', x_clean)
    
    # Eliminar apostrofes
    x_clean = re.sub(r"'", '', x_clean)
    # Separar numeros y palabras
    x_clean = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', x_clean)
    # Separar numeros y palabras
    x_clean = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', x_clean)
    # Eliminar números
    x_clean = re.sub(r'\b\d+\b', '', x_clean)
    # Eliminar palabras 1/2 caracteres 
    x_clean = re.sub(r'\b\w{1,2}\b', '', x_clean)
    # Eliminar caracteres especiales
    x_clean = re.sub(r'\W', ' ', x_clean)
    # Eliminar caracteres simples
    x_clean = re.sub(r'\s+[a-zA-Z]\s+', ' ', x_clean)
    # Eliminar caracteres simples del inicio
    x_clean = re.sub(r'\^[a-zA-Z]\s+', ' ', x_clean)
    # Eliminar múltiples espacios por uno solo
    x_clean = re.sub(r'\s+', ' ', x_clean, flags=re.I)
    # Convertir todo el texto a minúsculas
    x_clean = x_clean.lower()
    
    
    return(x_clean)
    

# Tokenizing and removing stopwords
def word_token_sinstop(x):
    import nltk
    stopwords = nltk.corpus.stopwords.words('english')
    x_token = nltk.tokenize.word_tokenize(x)
    x_token= [word for word in x_token if word not in stopwords]
    
    return(x_token)

# Tokenizing without removing stopwords
def word_token(x):
    import nltk
    x_token = nltk.tokenize.word_tokenize(x)
    
    return(x_token)

# Cleaning chorus
def limpieza_chorus(x):
    import re
    x = re.sub(r'\bchorus\b', '', x, flags=re.IGNORECASE)
    return x

# Lexical richeness
 def riqueza_lexica(tokens):
    tokens_conjunto=set(tokens)
    palabras_totales=len(tokens)
    palabras_diferentes=len(tokens_conjunto)
    riqueza_lexica=palabras_diferentes/palabras_totales
    return riqueza_lexica