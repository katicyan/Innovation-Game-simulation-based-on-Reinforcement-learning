import argparse
# Example: parse --name and --repeat from command line
parser = argparse.ArgumentParser(description="Simple argparse example")
parser.add_argument("--name", type=str, default="World", help="Name to greet")
parser.add_argument("--repeat", type=int, default=1, help="How many times to print")
parser.add_argument("--weather",type=str,default="cloudy",help="Weather condition")
# For normal Python scripts, use:
args = parser.parse_args()
note = {
    "cloudy": "It's a bit gloomy outside.",
    "sunny": "It's a bright and sunny day! If need please use sunscreen.",
    "rainy": "Don't forget your umbrella",
}
if args.weather not in note:
    intro = "Variable weather conditions, please pay attention to the weather forecast."
else:
    intro  = note[args.weather]
for _ in range(args.repeat):

    print(f"Hello, {args.name}!\n The weather today is {args.weather}.\n {intro}")