# HOW POPULAR WOULD MY SONG BE?
In this project we will try to answer to this question, trying to predict the popularity of songs with lyrics basing the analysis on the audio features and the lyrics of the song.
Due to the fact that selecting the initial data manually and choosing the songs for the study would have taken much longer, we decided to start by looking for an existing dataframe with
Spotify track id’s in order to be able to work with the Spotify Track Id as Primary Key for extracting new data if necessary later.
In this sense, the initial data collects 30000 Spotify songs with several audio features that Spotify already provides for its tracks and that may have been obtained using Spotify API. 
You will find it within our documentation but it’s also available at https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs?select=spotify_songs.csv

DOCUMENTATION:
All the data used in the project. You will find a description of the df_final features, that will be the ones used in our model.
- DATAFRAMES:

df_initial_32833.csv: Starting dataframe with 32833 instances. Features include 'track_id', 'track_name', 'track_artist', 'track_popularity', 'track_album_id', 'track_album_name','track_album_release_date', ‘playlist_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre',
'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness’, ‘acousticness', 'instrumentalness','liveness', 'valence', 'tempo', 'duration_ms’.

df_songs_lang_lyrics_8072.csv: Dataframe with 8072 instances, including previous features, language of lyrics, lyrics and artist popularity. Songs with English lyrics only and after deleting duplicates.

df_songs_resampled_7637.csv: Similar to the previous dataframe but after resampling for normalizing the distribution of ‘track_popularity’. It has 7637 instances and it is the definitive number of instances we will be working with.

df_lyrics.csv: It is the lyrics dataframe, it has 7637 instances and features include the clean
lyrics of the song and other lyric features which origin is explained in the 4th notebook.

df_final.csv: It is the dataframe we will be using for machine learning, it has 7637 instances and features include (This csv is too big for Github so it is not uploaded!):
- ‘track_id’: Song unique ID
- ‘track_name’: Song name
- ‘track_artist’: Song artist
- ‘track_popularity’: Song popularity (0-100) where higher is better.
- ‘track_album_id’: Album unique ID
- ‘track_album_name’: Song album name
- ‘playlist_name’: Name of playlist
- ‘playlist_id’: Playlist ID
- ‘playlist_genre’: Playlist genre
- ‘playlist_subgenre’: Playlist Subgenre
- ‘danceability’: Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- ‘energy’: A measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- ‘key’: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- ‘loudness’: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
- ‘mode’: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- ‘speechiness’: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- ‘acousticness’: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- ‘instrumentalness’: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- ‘liveness’: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- ‘valence’: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- ‘tempo’: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
- ‘duration_ms’: Duration of song in milliseconds.
- ‘release_year’: Year the song was released.
- ‘idioma’: Language of the lyrics.
- ‘lyrics’: Complete lyrics of the song.
- ‘artist_popularity’: The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.
- ‘track_album_release_month’: Month the song was released.
- ‘clean_lyrics’: Lyrics after a text cleaning function.
- ‘len_words’: Total number of words in clean_lyrics without stop words.
- ‘riq_lex’: Lexical richness of the lyrics without stop words.
- ‘most1’: Most frequent word in the song without stop words.
- ‘most1_freq’: Number of occurrences of most1 in the song.
- ‘most2’: Second most frequent word in the song without stop words.
- ‘most2_freq’: Number of occurrences of most2 in the song.
- ‘most3’: Third most frequent word in the song without stop words.
- ‘most3_freq’: Number of occurrences of most3 in the song.
- ‘less1’: Less frequent word in the song without stop words.
- ‘less1_freq’: Number of occurrences of less1 in the song.
- ‘less2’: Second less frequent word in the song without stop words.
- ‘less2_freq’: Number of occurrences of less2 in the song.
- ‘less3’: Third less frequent word in the song without stop words.
- ‘less3_freq’: Number of occurrences of less3 in the song.
- ‘len_ws’: Total number of words in clean_lyrics with stop words.
- ‘riq_lex_ws’: Lexical richness of the lyrics with stop words.
- ‘MostFreqPercentage’: Percentage of presence in each song of the most frequent words within all the text.
- ‘explicit’: Percentage of explicit vocabulary.
- ‘non_explicit’: Percentage of non explicit vocabulary.
- ‘has_intro’: 0 if the song does not have an intro, 1 if it has an intro.
- ‘has_outro’: 0 if the song does not have an intro, 1 if it has an intro.
- ‘chorus_count’: Count of chorus in the song.
- ‘prechorus_count’: Count of pre-chorus in the song.
- ‘verse_count’: Count of verses in the song.
- ‘coef_chorus/len’: Coefficient between chorus_count and len_words
- ‘verse_chorus/len’: Coefficient between chorus_count and len_words
- ‘len_estribillo_clean’: Total characters of the chorus. If the song doesn’t have a chorus, this value would be 300 or less.
- ‘topic1’: First topic label of the song.
- ‘topic2’: Second topic label of the song.
- ‘topic3’: Third topic label of the song.

- NOTEBOOKS:
1. Exploring raw data: Brief analysis of the df_initial.
2. Obtaining language_lyrics_artistpop: Eliminating duplicates. Using Spotify API for obtaining the artist_popularity, the MusixMatch API for obtaining the language of the lyrics and web scrapping for obtaining the lyrics of the songs in English. Finally exporting df_songs_lang_lyrics_8072.csv.
4. EDA and resampling: Exploratory analysis of features and resampling for normal distribution of Y (‘track_popularity’). Finally exporting df_songs_resampled_7637.csv.
5. Obtaining lyrics dataframe: NLP of lyrics, obtaining lyrics features. Finally exporting df_lyrics.csv.
6. Chorus and topic labelling: Merging the dataframe with the audio features and the one with the lyrics features. LDA Topic modeling and getting 3 topic labels for each song. Finally exporting df_final.csv.
7. Models: Final analysis of features and machine learning with XGBoost Regression.

- FUNCTIONS
- Funciones.py: A Python document where you can find the most important functions we will be using and importing. Functions include: limpieza(x), word_token_sinstop(x), word_token(x), limpieza_chorus(x) and riqueza_lexica(x).
