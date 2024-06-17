import os
import openai
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 기본적으로 설치된 폰트를 사용합니다.
fontprop = fm.FontProperties(fname=fm.findSystemFonts(fontpaths=None, fontext='ttf')[0])
plt.rc('font', family='DejaVu Sans')


# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
data = None

def load_data(file_path):
    global data
    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        data = None

def calculate_improvement(data):
    improvements = []
    for name, group in data.groupby('학생'):
        initial_scores = group.groupby('과목')['점수'].first()
        final_scores = group.groupby('과목')['점수'].last()
        improvement = final_scores - initial_scores
        total_improvement = improvement.sum()
        improvements.append((name, total_improvement, improvement))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    return improvements

def find_student_needing_most_improvement(data):
    lowest_scores = []
    for name, group in data.groupby('학생'):
        average_score = group['점수'].mean()
        lowest_scores.append((name, average_score))
    
    lowest_scores.sort(key=lambda x: x[1])
    return lowest_scores[0]

def find_highest_average_student(data):
    highest_scores = []
    for name, group in data.groupby('학생'):
        average_score = group['점수'].mean()
        highest_scores.append((name, average_score))
    
    highest_scores.sort(key=lambda x: x[1], reverse=True)
    return highest_scores[0]

def find_lowest_average_student(data):
    lowest_scores = []
    for name, group in data.groupby('학생'):
        average_score = group['점수'].mean()
        lowest_scores.append((name, average_score))
    
    lowest_scores.sort(key=lambda x: x[1])
    return lowest_scores[0]

def find_subject_with_lowest_average(data):
    avg_scores = data.groupby('과목')['점수'].mean()
    return avg_scores.idxmin(), avg_scores.min()

def find_subject_with_highest_average(data):
    avg_scores = data.groupby('과목')['점수'].mean()
    return avg_scores.idxmax(), avg_scores.max()

def find_students_below_threshold(data, threshold):
    below_threshold = data[data['점수'] < threshold]
    return below_threshold['학생'].unique().tolist()

def find_students_above_threshold(data, threshold):
    above_threshold = data[data['점수'] > threshold]
    return above_threshold['학생'].unique().tolist()

def get_gpt4_response(data_str, question, initial_response):
    # LangChain 설정
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, max_tokens=700)

    template = """
    You are an AI assistant that analyzes student scores and provides detailed insights based on the data. Answer questions related to student performance, improvement, and areas needing additional attention in a detailed and thorough manner.

    Student scores data:
    {data}

    Question: {question}

    Initial Response: {initial_response}

    Detailed Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["data", "question", "initial_response"])
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(data=data_str, question=question, initial_response=initial_response)

def find_highest_and_lowest_scores(data):
    highest_score = data['점수'].max()
    lowest_score = data['점수'].min()
    highest_student = data[data['점수'] == highest_score]['학생'].values[0]
    lowest_student = data[data['점수'] == lowest_score]['학생'].values[0]
    return highest_student, highest_score, lowest_student, lowest_score

def calculate_subject_average(data, subject):
    subject_data = data[data['과목'] == subject]
    avg_score = subject_data['점수'].mean()
    return avg_score

def calculate_student_average(data, student):
    student_data = data[data['학생'] == student]
    avg_score = student_data['점수'].mean()
    return avg_score

def find_student_with_highest_variance(data):
    variances = []
    for name, group in data.groupby('학생'):
        variance = group['점수'].var()
        variances.append((name, variance))
    
    variances.sort(key=lambda x: x[1], reverse=True)
    return variances[0]

def find_top_student_per_subject(data):
    top_students = data.loc[data.groupby('과목')['점수'].idxmax()]
    return top_students

def find_student_with_lowest_variance(data):
    variances = []
    for name, group in data.groupby('학생'):
        variance = group['점수'].var()
        variances.append((name, variance))
    
    variances.sort(key=lambda x: x[1])
    return variances[0]

def find_lowest_student_per_subject(data):
    lowest_students = data.loc[data.groupby('과목')['점수'].idxmin()]
    return lowest_students

def find_student_best_subject(data, student):
    student_data = data[data['학생'] == student]
    best_subject = student_data.loc[student_data['점수'].idxmax()]
    return best_subject['과목'], best_subject['점수']

def find_student_worst_subject(data, student):
    student_data = data[data['학생'] == student]
    worst_subject = student_data.loc[student_data['점수'].idxmin()]
    return worst_subject['과목'], worst_subject['점수']

def find_student_score_distribution(data, student):
    student_data = data[data['학생'] == student]
    return student_data

def analyze_data(data, question):
    if data is None:
        return "데이터를 로드하지 못했습니다. 파일을 다시 업로드해 주세요.", None
    img_url = None
    data_str = data.to_csv(index=False)

    if "성적이 가장 많이 향상된 학생" in question:
        improvements = calculate_improvement(data)
        top_student = improvements[0]
        initial_response = f"{top_student[0]}는 총 {top_student[1]}점 향상했습니다. 세부 사항: {top_student[2].to_dict()}"
        
        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.bar(top_student[2].index, top_student[2].values, color='skyblue')
        plt.title(f'{top_student[0]}의 성적 향상', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수 향상', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()
        
        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "학생별 평균 점수" in question:
        student = question.split(" ")[0]
        avg_score = calculate_student_average(data, student)
        initial_response = f"{student} 학생의 전체 평균 점수는 {avg_score:.2f}점입니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.bar(student_data['과목'], student_data['점수'], color='lightblue')
        plt.title(f'{student}의 과목별 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/student_avg_score_chart.png')
        plt.close()

        img_url = 'static/student_avg_score_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "학생별 과목 최저 점수" in question:
        student = question.split(" ")[0]
        subject, score = find_student_worst_subject(data, student)
        initial_response = f"{student} 학생의 최저 점수는 {subject} 과목에서 {score}점입니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.bar(student_data['과목'], student_data['점수'], color='red')
        plt.title(f'{student}의 과목별 최저 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/student_worst_subject_chart.png')
        plt.close()

        img_url = 'static/student_worst_subject_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "학생의 전체 성적 분포" in question:
        student = question.split(" ")[0]
        student_data = find_student_score_distribution(data, student)
        initial_response = f"{student} 학생의 전체 성적 분포는 다음과 같습니다: {student_data['점수'].describe().to_dict()}"

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.hist(student_data['점수'], bins=10, color='orange')
        plt.title(f'{student}의 전체 성적 분포', fontproperties=fontprop)
        plt.xlabel('점수', fontproperties=fontprop)
        plt.ylabel('과목 수', fontproperties=fontprop)
        plt.savefig('static/student_score_distribution_chart.png')
        plt.close()

        img_url = 'static/student_score_distribution_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "학생별 과목 최고 점수" in question:
        student = question.split(" ")[0]
        subject, score = find_student_best_subject(data, student)
        initial_response = f"{student} 학생의 최고 점수는 {subject} 과목에서 {score}점입니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.bar(student_data['과목'], student_data['점수'], color='blue')
        plt.title(f'{student}의 과목별 최고 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/student_best_subject_chart.png')
        plt.close()

        img_url = 'static/student_best_subject_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "과목별 최저 성적 학생" in question:
        lowest_students = find_lowest_student_per_subject(data)
        initial_response = f"과목별 최저 성적 학생은 다음과 같습니다:\n{lowest_students[['과목', '학생', '점수']].to_string(index=False)}"

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        for subject in lowest_students['과목'].unique():
            plt.bar(subject, lowest_students[lowest_students['과목'] == subject]['점수'], label=lowest_students[lowest_students['과목'] == subject]['학생'].values[0])
        plt.title('과목별 최저 성적 학생', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.legend()
        plt.savefig('static/lowest_student_per_subject_chart.png')
        plt.close()

        img_url = 'static/lowest_student_per_subject_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "성적이 가장 균일한 학생" in question:
        student, variance = find_student_with_lowest_variance(data)
        initial_response = f"{student} 학생의 성적이 가장 균일합니다. 분산 값은 {variance:.2f}입니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.plot(student_data['과목'], student_data['점수'], marker='o', color='green')
        plt.title(f'{student}의 성적 균일도', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/student_uniformity_chart.png')
        plt.close()

        img_url = 'static/student_uniformity_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "과목별 최고 성적 학생" in question:
        top_students = find_top_student_per_subject(data)
        initial_response = f"과목별 최고 성적 학생은 다음과 같습니다:\n{top_students[['과목', '학생', '점수']].to_string(index=False)}"

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        for subject in top_students['과목'].unique():
            plt.bar(subject, top_students[top_students['과목'] == subject]['점수'], label=top_students[top_students['과목'] == subject]['학생'].values[0])
        plt.title('과목별 최고 성적 학생', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.legend()
        plt.savefig('static/top_student_per_subject_chart.png')
        plt.close()

        img_url = 'static/top_student_per_subject_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "성적 변동이 큰 학생" in question:
        student, variance = find_student_with_highest_variance(data)
        initial_response = f"{student} 학생의 성적 변동이 가장 큽니다. 분산 값은 {variance:.2f}입니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.plot(student_data['과목'], student_data['점수'], marker='o', color='orange')
        plt.title(f'{student}의 성적 변동', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/student_variance_chart.png')
        plt.close()

        img_url = 'static/student_variance_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    elif "최고 점수와 최저 점수" in question:
        highest_student, highest_score, lowest_student, lowest_score = find_highest_and_lowest_scores(data)
        initial_response = f"최고 점수는 {highest_student} 학생이 {highest_score}점을 받았습니다. 최저 점수는 {lowest_student} 학생이 {lowest_score}점을 받았습니다."

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.bar(['최고 점수', '최저 점수'], [highest_score, lowest_score], color=['blue', 'red'])
        plt.title('최고 점수와 최저 점수', fontproperties=fontprop)
        plt.xlabel('점수 종류', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/highest_lowest_scores_chart.png')
        plt.close()

        img_url = 'static/highest_lowest_scores_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "특정 과목 평균 점수" in question:
        subject = question.split(" ")[-1]
        avg_score = calculate_subject_average(data, subject)
        initial_response = f"{subject} 과목의 평균 점수는 {avg_score:.2f}점입니다."

        # 그래프 생성
        subject_data = data[data['과목'] == subject]
        plt.figure(figsize=(10, 6))
        plt.hist(subject_data['점수'], bins=10, color='purple')
        plt.title(f'{subject} 과목의 점수 분포', fontproperties=fontprop)
        plt.xlabel('점수', fontproperties=fontprop)
        plt.ylabel('학생 수', fontproperties=fontprop)
        plt.savefig('static/subject_avg_score_chart.png')
        plt.close()

        img_url = 'static/subject_avg_score_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "보충학습이 필요한 학생" in question:
        student, avg_score = find_student_needing_most_improvement(data)
        subject_needs = data[data['학생'] == student].groupby('과목')['점수'].mean().idxmin()
        initial_response = f"{student}는 평균 {avg_score:.2f}점으로 보충학습이 필요합니다. 특히 {subject_needs} 과목이 필요합니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.bar(student_data['과목'], student_data['점수'], color='salmon')
        plt.title(f'{student}의 과목별 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "성적이 가장 우수한 학생" in question or "가장 높은 평균 점수" in question:
        student, avg_score = find_highest_average_student(data)
        initial_response = f"{student}는 평균 {avg_score:.2f}점으로 가장 우수한 학생입니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.bar(student_data['과목'], student_data['점수'], color='gold')
        plt.title(f'{student}의 과목별 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "반 평균 점수" in question or "우리 반 평균" in question:
        avg_scores = data.groupby('과목')['점수'].mean()
        overall_avg_score = data['점수'].mean()
        initial_response = f"우리 반 전체 평균 점수는 {overall_avg_score:.2f}점입니다. 과목별 평균 점수는 다음과 같습니다: {avg_scores.to_dict()}"

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.bar(avg_scores.index, avg_scores.values, color='green')
        plt.title('과목별 반 평균 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('평균 점수', fontproperties=fontprop)
        plt.savefig('static/average_scores_chart.png')
        plt.close()

        img_url = 'static/average_scores_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url


    elif "성적이 가장 낮은 학생" in question or "가장 낮은 평균 점수" in question:
        student, avg_score = find_lowest_average_student(data)
        initial_response = f"{student}는 평균 {avg_score:.2f}점으로 성적이 가장 낮은 학생입니다."

        # 그래프 생성
        student_data = data[data['학생'] == student]
        plt.figure(figsize=(10, 6))
        plt.bar(student_data['과목'], student_data['점수'], color='red')
        plt.title(f'{student}의 과목별 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "가장 어려워하는 과목" in question or "가장 점수가 낮은 과목" in question:
        subject, avg_score = find_subject_with_lowest_average(data)
        initial_response = f"가장 점수가 낮은 과목은 {subject}로 평균 점수는 {avg_score:.2f}입니다."

        # 그래프 생성
        subject_data = data[data['과목'] == subject]
        plt.figure(figsize=(10, 6))
        plt.hist(subject_data['점수'], bins=10, color='salmon')
        plt.title(f'{subject} 과목의 점수 분포', fontproperties=fontprop)
        plt.xlabel('점수', fontproperties=fontprop)
        plt.ylabel('학생 수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "가장 잘하는 과목" in question or "가장 점수가 높은 과목" in question:
        subject, avg_score = find_subject_with_highest_average(data)
        initial_response = f"가장 점수가 높은 과목은 {subject}로 평균 점수는 {avg_score:.2f}입니다."

        # 그래프 생성
        subject_data = data[data['과목'] == subject]
        plt.figure(figsize=(10, 6))
        plt.hist(subject_data['점수'], bins=10, color='gold')
        plt.title(f'{subject} 과목의 점수 분포', fontproperties=fontprop)
        plt.xlabel('점수', fontproperties=fontprop)
        plt.ylabel('학생 수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "기초학력 미달 학생" in question or "기준 점수 이하" in question:
        threshold = 40  # 기준 점수
        students_below = find_students_below_threshold(data, threshold)
        initial_response = f"기초학력 미달 학생: {', '.join(students_below)}"
        if not students_below:
            initial_response = "기초학력 미달 학생이 없습니다."
        
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", None

    elif "우수 학생" in question or "기준 점수 이상" in question:
        threshold = 90  # 기준 점수
        students_above = find_students_above_threshold(data, threshold)
        initial_response = f"우수 학생: {', '.join(students_above)}"
        if not students_above:
            initial_response = "우수 학생이 없습니다."
        
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", None

    elif "학생의 과목별 점수" in question:
        name = question.split(" ")[0]
        student_data = data[data['학생'] == name]
        initial_response = f"{name}의 과목별 점수: {student_data[['과목', '점수']].to_dict(orient='records')}"

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.bar(student_data['과목'], student_data['점수'], color='lightblue')
        plt.title(f'{name}의 과목별 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('점수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "반 평균 점수" in question:
        avg_scores = data.groupby('과목')['점수'].mean()
        initial_response = f"반 평균 점수: {avg_scores.to_dict()}"

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.bar(avg_scores.index, avg_scores.values, color='green')
        plt.title('과목별 반 평균 점수', fontproperties=fontprop)
        plt.xlabel('과목', fontproperties=fontprop)
        plt.ylabel('평균 점수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url

    elif "특정 과목 성적" in question:
        subject = question.split(" ")[-1]
        subject_data = data[data['과목'] == subject]
        initial_response = f"{subject} 과목의 점수 분포: {subject_data['점수'].describe().to_dict()}"

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.hist(subject_data['점수'], bins=10, color='purple')
        plt.title(f'{subject} 과목의 점수 분포', fontproperties=fontprop)
        plt.xlabel('점수', fontproperties=fontprop)
        plt.ylabel('학생 수', fontproperties=fontprop)
        plt.savefig('static/improvement_chart.png')
        plt.close()

        img_url = 'static/improvement_chart.png'
        gpt4_response = get_gpt4_response(data_str, question, initial_response)
        return f"{initial_response}\n\n{gpt4_response}", img_url
    
    
    # LangChain 설정
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, max_tokens=700)

    template = """
    You are an AI assistant that analyzes student scores and provides detailed insights based on the data. Answer questions related to student performance, improvement, and areas needing additional attention in a detailed and thorough manner.

    Student scores data:
    {data}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["data", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)

    answer = chain.run(data=data_str, question=question)
    return answer, None


@app.route('/', methods=['GET', 'POST'])
def index():
    global data
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('data', file.filename)
            file.save(file_path)
            load_data(file_path)
            return redirect(url_for('analyze'))
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    global data
    if request.method == 'POST':
        question = request.form['question']
        answer, img_url = analyze_data(data, question)
        return render_template('analyze.html', question=question, answer=answer, img_url=img_url)
    return render_template('analyze.html')

if __name__ == '__main__':
    app.run(debug=True)
