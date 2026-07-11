import encrypt as e

KEYSEED = 948729038745908237459734957230573098475092
key = e.buildkey(KEYSEED)
image = "fuhrer"
e.encrypt_image(f"images/{image}.jpg", key, f"images/{image}.enc")
e.decrypt_image(f"images/{image}.enc", key, f"images/{image}_dec.jpg")
