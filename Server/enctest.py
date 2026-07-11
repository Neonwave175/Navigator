import encrypt as e

KEYSEED = 948729038745908237459734957230573098475092
key = e.buildkey(KEYSEED)
images = ["Concorde", "Jensen", "Linus", "Newt", "SteveJobs"]
for i in images:
    e.encrypt_image(f"images/{i}.jpg", key, f"encrypted/{i}.enc")
    e.decrypt_image(f"encrypted/{i}.enc", key, f"decrypted/{i}.jpg")
