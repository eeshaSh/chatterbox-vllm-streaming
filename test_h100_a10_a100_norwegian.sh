curl -N -X POST "http://20.163.2.63:4123/tts" -d "text=Jeg heter Ece og er tolv år gammel. Hver morgen våkner jeg klokken syv, spiser frokost og går på skolen." -d "language_id=no" -d "format=pcm" -d "chunk_size=30" -d "diffusion_steps=5"  | ffplay -f s16le -ar 24000 -nodisp -autoexit -

curl -N -X POST "http://23.100.39.56:4123/tts" -d "text=Jeg heter Ece og er tolv år gammel. Hver morgen våkner jeg klokken syv, spiser frokost og går på skolen." -d "language_id=no" -d "format=pcm" -d "chunk_size=30" -d "diffusion_steps=5"  | ffplay -f s16le -ar 24000 -nodisp -autoexit -

curl -N -X POST "http://20.168.112.102:4123/tts" -d "text=Jeg heter Ece og er tolv år gammel. Hver morgen våkner jeg klokken syv, spiser frokost og går på skolen." -d "language_id=no" -d "format=pcm" -d "chunk_size=15" -d "diffusion_steps=5"  | ffplay -f s16le -ar 24000 -nodisp -autoexit -
