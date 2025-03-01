from openai import OpenAI

client = OpenAI(api_key="XYZ")

with open("patterns/patterns2.json", "r") as f:
    data = f.read()

prompt = 'analyse this tracr model layer by layer. compare provided data with known patterns from other simple tasks and at the end predict top 1 task it propably preforms'
for _ in range(8):
    completion = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n{data}"
            }
        ]
    )

    print(completion.choices[0].message)
