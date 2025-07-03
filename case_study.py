import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import pprint as pp
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import json
import faiss
# device = torch.device("cuda:0") 
import numpy as np
from utils import Agent
import re
import ast
import pandas as pd
import time
from collections import defaultdict
from utils import read_jsonlines, multi_label_metric, ddi_rate_score, Voc, EHRTokenizer
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import random



def getlist(content): 
    if type(content) == str:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            transformed_content = match.group(0)
            content = ast.literal_eval(transformed_content)
        else:
            print("List not found.")
    return content

def get_conflict_dict(assign_list):
    result = {}
    for item in assign_list:
        expert, numbers_str = item.split(":")
        expert = expert.strip()
        numbers_str = numbers_str.strip()
        if len(numbers_str) > 0:
            parts = numbers_str.split(",")
            numbers = []
            for part in parts:
                part = part.strip()  
                if part.isdigit():   
                    numbers.append(int(part))
                else:
                    part = part[-1]
                    print(part)
                    numbers.append(int(part))
            result[expert] = numbers
    return result


def generate_advices_from_single(single_chat_list, agent_specialists, conflicts_splited, goal_rel_med, goals, patient_condition):
    print("Single advice generating...")
    advices_from_single = "Recommendations addressing the drug conflicts:"
    for conflict_idx, specialist_item in single_chat_list.items():
        for agent_specialist in agent_specialists:
            if agent_specialist.role == specialist_item[0]:
                prompt = f'''
                Review the patient's condition <{patient_condition}>, current clinical goals <{goals}>, and current treatment plan <{goal_rel_med}>, and here are the identified conflicts relevant to your expertise <{conflicts_splited[conflict_idx-1]}>.
                Provide a recommendation related to the conflict regarding your field of expertise or that may exacerbate the condition in your field. when you suggest a revised treatment plan, please consider all comorbidities, risks, clinical goals, but keep the number of medications prescribed at a reasonable number. Pay attention to timing: let us know if the dose of a drug that is started or stopped should be given or removed gradually and specify how, but do it only if it is required. Remember that not all conflicts can always be solved.
                Clearly explain the reasoning behind your recommendation and any adjustments to the treatment plan.
                Note: Only output the adjustment with reason.
                Output format: 
                1. **Adjustments:**
                '''
                advice_from_single = f"Advive from {agent_specialist.role} regarding conflict {conflict_idx}: \n" + agent_specialist.chat(prompt)
                print(advice_from_single)
                advices_from_single += "\n" + advice_from_single + "\n---"
    return advices_from_single

#chat room
def judge_consesus(agent_mediator, advices_from_multi, agent_mediator_mes_text):
    agent_mediator.messages = agent_mediator.messages[:1] 
    consesus_comparison = "\n---\n".join(
        [f"{item}" for index, item in enumerate(advices_from_multi)]
    )
    prompt = f'''
    In this round of discussions, clinical experts proposed advices as follows: <{consesus_comparison}>.
    The definition of consensus is that several clinical experts have exactly the same opinion.
    Your job is to determine if they have reached a consensus.
    Only respond with 'yes' or 'no'.

    Example 1:
    Specialist 1: Delete drug A. Add drug B.
    Specialist 2: Delete drug A. Add drug C.
    Consensus output: 'no'

    Example 2:
    Specialist 1: Delete drug A. Add drug B.
    Specialist 2: Delete drug A. Add drug B.
    Consensus output: 'yes'

    '''
    consensus_reached = agent_mediator.chat(prompt)
    agent_mediator_mes_text += agent_mediator.messages[1:]
    return consensus_reached, agent_mediator_mes_text

def generate_first_round_advices(prompt, multi_chat_members):    
    initial_advices = [] 
    print(multi_chat_members)
    for multi_chat_member in multi_chat_members:
        multi_chat_member.messages = multi_chat_member.messages[:1] 
        advice = multi_chat_member.chat(prompt)
        print(f"Advice from {multi_chat_member.role}:", advice)
        advice_from_multi = f"{multi_chat_member.role} proposed:" + advice
        initial_advices.append(advice_from_multi)
    return initial_advices

def generate_next_round_advices(multi_chat_members, advices_from_multi):    
    next_round_advices = [] 
    other_advices = []
    for i in range(len(advices_from_multi)):
        other_advice = "".join([advices_from_multi[j] if j != i else "" for j in range(len(advices_from_multi))])
        other_advices.append(other_advice)
    for index, multi_chat_member in enumerate(multi_chat_members):
        advice = multi_chat_member.chat(other_advices[index] + '''
        Do you think there are any recommendations among these that are more reasonable than the ones you proposed? 
        You can also put forward new modification recommendation.
        ''')
        print(f"Advice from {multi_chat_member.role}:", advice)
        advice_from_multi = f"{multi_chat_member.role} proposed:" + advice
        next_round_advices.append(advice_from_multi)
    return next_round_advices

def consensus_process(agent_mediator, multi_chat_members, initial_advices):
    advices_from_multi = initial_advices  
    max_rounds = 5  
    current_round = 0
    agent_mediator_mes_text = []
    while current_round < max_rounds:
        consensus_reached, agent_mediator_mes_text = judge_consesus(agent_mediator, advices_from_multi, agent_mediator_mes_text)
        if "yes" in consensus_reached.lower():
            consensus_prompt = "What is the final consensus?"
            consensus_advice = agent_mediator.chat(consensus_prompt)
            print("Final consensus reached:", consensus_advice)
            return consensus_advice, current_round, agent_mediator_mes_text 
        elif "no" in consensus_reached.lower():
            current_round += 1
            print(f"Round {current_round}: No consensus reached. Generating next round advices...")
            advices_from_multi = generate_next_round_advices(multi_chat_members, advices_from_multi)
        else:
            print("Unexpected response from judge_consesus:", consensus_reached)
            return "No consensus"  
    if current_round >= max_rounds:
        print("Max rounds reached. No consensus achieved.")
        consensus_prompt = "What is the most possible final consensus?"
        consensus_advice = agent_mediator.chat(consensus_prompt)
        print("Final possible consensus:", consensus_advice)
        return consensus_advice, current_round, agent_mediator_mes_text
    
def generate_consensus_advice(conflict_idx, specialists_item, agent_specialists, conflicts_splited, goal_rel_med, goals, patient_condition):
    multi_chat_members = [] 
    for specialist_item in specialists_item: 
        for agent_specialist in agent_specialists:
            if agent_specialist.role == specialist_item:
                agent_specialist.messages[:2] 
                multi_chat_members.append(agent_specialist)
    prompt = f'''
                Review the patient's condition <{patient_condition}>, current clinical goals <{goals}>, and current treatment plan <{goal_rel_med}>, and here are the identified conflicts relevant to your expertise <{conflicts_splited[conflict_idx-1]}>.
                Provide a recommendation related to the conflict regarding your field of expertise or that may exacerbate the condition in your field. when you suggest a revised treatment plan, please consider all comorbidities, risks, clinical goals, but keep the number of medications prescribed at a reasonable number. Pay attention to timing: let us know if the dose of a drug that is started or stopped should be given or removed gradually and specify how, but do it only if it is required. Remember that not all conflicts can always be solved.
                Clearly explain the reasoning behind your recommendation and any adjustments to the treatment plan.
                Note: Only output the adjustment with reason.
                Output format: 
                1. **Adjustments:**
                '''
    initial_advices = generate_first_round_advices(prompt, multi_chat_members) 
    MEDIATOR_INSTRUCT = "You are a professional mediator in a chat room where several clinical experts are proposing their suggestions on drug interactions. Your job is to determine if they have reached a consensus, and provide final discussion result to the medication plan adjustment."
    agent_mediator = Agent(MEDIATOR_INSTRUCT, 'mediator') 
    consensus_advice, current_round, agent_mediator_mes_text = consensus_process(agent_mediator, multi_chat_members, initial_advices)
    return consensus_advice, current_round, agent_mediator_mes_text



def generate_advices_from_multi(multi_chat_list, agent_specialists, conflicts_splited, goal_rel_med, goals, patient_condition):
    print("Creating a chatroom")
    advices_from_multi = ""
    advices_from_multi_rounds = []
    advices_from_multi_text = []
    for conflict_idx, specialists_item in multi_chat_list.items():
        print(type(conflict_idx))
        print(f"Conflict{conflict_idx}chatroom")
        consensus_advice, current_round, agent_mediator_mes_text = generate_consensus_advice(conflict_idx, specialists_item, agent_specialists, conflicts_splited, goal_rel_med, goals, patient_condition)
        advices_from_multi += f"Recommendations for conflict {conflict_idx}: " + consensus_advice + "\n---"
        advices_from_multi_rounds.append(current_round)
        advices_from_multi_text.append(agent_mediator_mes_text)
    return advices_from_multi, advices_from_multi_rounds, advices_from_multi_text


def create_agent_specialists(organized_MDT_plan, RESTRICT_INSTRUCT):
    if type(organized_MDT_plan) == str:
        specialists  = ast.literal_eval(organized_MDT_plan)
    else:
        specialists  = organized_MDT_plan
    if(len(specialists) < 2): 
        specialists = ["practitioner"]
    print(specialists)
    SPECIALIST_INSTRUCT = "You are a <SUBJECT> specialist, your job is to provide expertise in your specific field to contribute to the treatment plan based on the patient's condition."
    agent_specialists = []
    for index, specialist in enumerate(specialists):
        specialist_instruct_new = SPECIALIST_INSTRUCT.replace("<SUBJECT>", specialist) + RESTRICT_INSTRUCT
        agent_name = f"{specialist}"  
        agent_specialist = Agent(specialist_instruct_new, agent_name)
        agent_specialists.append(agent_specialist)
        print(f"Created agent: {agent_name} for {specialist}")
    return agent_specialists


def conflict_revise(specialist_advices, agent_GP, type="test"): 
    if type == "analyze":
        agent_GP.messages = agent_GP.messages[:4]
    
    revise_prompt = f'''
        Here are advices on conflicts from several specialists: <{specialist_advices}>.
       
        Your task is to:
        Basing on advices proposed by specialists, organize a revised comprehensive treatment plan.
        Explan how the plan meets the clinical goals while reduces the conflicts.  
    '''
    revised_result = agent_GP.chat(revise_prompt)
    return revised_result



if __name__ == "__main__":
    list_test_samples = [
        #case1
        {
            "input": """
            Patient case: Mrs. Williams is a 76 year old female, height 172cm, weight 70kg, BMI: 23.7
            Current problems: Transient Ischemic Attack (TIA), Duodenal Ulcer (DU)
            Current medications: 
            Aspirin: the patient is on aspirin for secondary prevention of stroke, due to TIA 13 years ago. 
            Nexium, a Proton Pump Inhibitor (PPI): because she had duodenal ulcer 4 years ago due to aspirin, the patient is on PPI to protect the duodenum and prevent ulcer bleeding,.
            New problem: Osteoporosis
            Management scenario: The patient presented recently with back pain. Earlier lumbosacral X-ray showed no vertebral fracture and the physician decided to follow primary care recommendations to evaluate risk for osteoporosis fractures, and thus ordered a DXA bone mineral density scan. 
            Osteoporosis was confirmed (DXA shows bone marrow density of -2.6. FRAX assessed and risk of second fracture is >20%). The recommended blood tests were ordered (including electrolytes, vitamin D, thyroid function, protein electrophoresis, CBC, and metabolic panel) to rule out additional reasons for secondary osteoporosis. All blood tests were normal. The patient does not have conditions that may be a secondary cause such as Diabetes or Celiac or a family history of them. 
            Another possible secondary cause of osteoporosis is Nexium (PPI). 
            """
        },
        #case2
         {
            "input": """
            Patient case: John is a 70 years old male, height 176 cm, weight 84 kg, BMI 26.5.
            Current problems: non-proteinuric chronic kidney disease (CKD) with severely decreased kidney function (eGFR of 35), significant anemia (hemoglobin level of 95), stable ferritin level of 110, no metabolic disturbances; hypertension (HTN).
            Current medications: erythropoiesis-stimulating agent (ESA), low-dose aspirin – to manage CKD; calcium channel blocker (CCB), diuretics – to manage HTN; ACE inhibitor (ACEI) – to manage both CKD and HTN.
            Current non-pharmacological treatment: lifestyle management to minimize the risk of cardiovascular disease (CVD) associated with CKD and to control HTN.
            New problem: atrial fibrillation (AFib)
            Management scenario: For last year John experienced multiple episodes of irregular heartbeat that resolved on its own. However, for last 2 days John is experiencing pronounced irregular heartbeat with the increasing intensity of the associated symptoms of breathlessness, dizziness, and chest discomfort. Upon admission to the ED, John is diagnosed with tachycardia and persistent, highly symptomatic acute, non-valvular AFib that is confirmed by standard ECG recording.
            John’s CHA2DS2-VASc score is greater than 2. 
            As a first line of urgent treatment John is administered intravenous heparin and his condition is stabilized with urgent direct-current cardioversion that results in significant improvement of the symptoms of AFib.
        """
        },
        
        #case3
        {
            "input": """
            Patient case: Mrs. Jones is a 67-year-old female, height 165cm, weight 65kg, BMI: 23.9
            Current problems: Recurrent Venous Thromboembolism (VTE)
            Current medications: 
            Warfarin (5mg / day): the patient is on long-term anticoagulation therapy for recurrent VTE; warfarin was chosen due to financial considerations. 
            New problem: Urinary Tract Infection (UTI)
            Management scenario: The patient presented recently with a strong and persistent urge to frequently urinate and describes a burning sensation when urinating.
            A preliminary diagnosis of cystitis UTI was confirmed by lab results. For the last episode of UTI, she was prescribed TMP/SMX (Bactrim, 2 double-strength tablets orally, twice daily for at least 3 days), which was well-tolerated by the patient. 
            The same treatment for the current episode of cystitis UTI is followed.
            """
        },
       
        #case4
        {
            "input": """
            Patient case: Mr. Grant is a 73 year old male, height 183 cm, weight 80 kg, BMI: 23.9.
            Current problems: MI and 2 months post Drug-eluting Stent surgery
            Current medications: Aspirin, Clopidogrel – dual antiplatelet therapy after the stent operation.
            New problem: Lung mass
            Management scenario: Patient had MI and the doctors decided to implant a drug-eluting stent. Accordingly, the patient was placed on dual anti-platelet therapy: aspirin + clopidogrel for 12 months.
            Two months after the stent implantation, the patient was diagnosed with a lung mass and surgery is indicated and cannot be postponed till dual antiplatelet therapy is completed.
        """
        },

    ]
    for index, list_test_sample in enumerate(list_test_samples):
        print("case", index+1)
        GP_INSTRUCT = "You are an experienced general practitioner."
        agent_GP = Agent(GP_INSTRUCT, 'practitioner') 
        patient_condition = list_test_sample['input'] 
        sets_of_prompt = f'''A patient with multimorbidity shows up with the current case <{patient_condition}>.
        Now, you are tasked to
        1) identify clinical goals for the patient that include prevention and management goals;
        2) identify the current interventions/medications that were prescribed for each clinical goal. There could be one or more medications per goal;
        3) Identify any conflicts in the current treatment plan, such as**drug-drug interactions** or **potential contraindications** or **potential exacerbations of a condition or its increased risk via a medication**, Sort the conflicts by severity;

        Output format:
        1. **Clinical Goals:**
        - Goal 1
        - Goal 2
        2. **Clinical Goals Related Medications:**
        - Goal 1: [Medication set 1]
        - Goal 2: [Medication set 2]
        3. **Potential Conflicts or Contraindications:**
        - Conflict 1: [Conflict Description 1]
        - Conflict 2: [Conflict Description 2]
        '''
        Goal_Med_Conf = agent_GP.chat(sets_of_prompt)
        print(Goal_Med_Conf)

        goal_rel_med_match = re.search(r"Clinical Goals Related Medications(.+?)Potential Conflicts or Contraindications", Goal_Med_Conf, re.DOTALL)
        goal_rel_med = goal_rel_med_match.group(1).strip() if goal_rel_med_match else ""
        print(goal_rel_med)

        conflicts_match = re.search(r"Potential Conflicts or Contraindications.*", Goal_Med_Conf, re.DOTALL)
        conflicts = conflicts_match.group(0).strip() if conflicts_match else ""
        print(conflicts)

        goals_match = re.search(r"Clinical Goals(.+?)Clinical Goals Related Medications", Goal_Med_Conf, re.DOTALL)
        goals = goals_match.group(1).strip() if goals_match else ""
        print(goals)
        conflicts_splited_match = re.search(r"Potential Conflicts or Contraindications.*", Goal_Med_Conf, re.DOTALL).group(0)
        pattern = r'(Conflict \d+)(.*?)(?=Conflict \d+|$)'
        matches = re.findall(pattern, conflicts_splited_match, re.DOTALL)
        conflicts_splited = []
        for match in matches:
            conflict_part = match[0]  
            content = match[1].strip() 
            conflicts_splited.append(conflict_part + content)  
        print(len(conflicts_splited),conflicts_splited)
        new_plan_gen = '''
        1) Propose a better comprehensive plan that will reduce the conflicts, but keep the number of medications prescribed at a reasonable size; pay attention to timing: let us know if the dose of a drug that is started or stopped should be given or removed gradually and specify how, but do it only if it is required;
        2) Explan how your plan meets the goals while reduces the conflicts without excessive polypharmacy;
        Output format:
        1. **New Plan:**
        2. **Clinical Goals Related Medications in New Plan:**
        - Goal 1: [Medication set 1]
        - Goal 2: [Medication set 2]
        3. **Potential Conflicts or Contraindications in New Plan:**
        - Conflict 1: [Conflict Description 1]
        - Conflict 2: [Conflict Description 2]
        '''
        new_conf = agent_GP.chat(new_plan_gen)
        print(new_conf)
        agent_GP.messages = agent_GP.messages[:4]
        recuit_MDT_prompt = f'''
        As an alternative to the **New Plan** you propsosed, review the initial clinical goals <{goals}>, their initial medications plan <{patient_condition}> and the initial conflicts <{conflicts}>, would you be interested in forming a small multidisciplinary team (MDT) to help you optimize therpy? Consider just the conflicts and propose which type of clinical expert is needed to solve the conflict. What types of medical experts would you assemble for the MDT? Please try to limit the size of the MDT. 
        Provide a clear summary that includes: 
        - **Specialists to Consult:** A list of relevant specialists needed to resolve these issues (e.g., Cardiologist, Pulmonologist, Nephrologist). 
        - **Assign conflicts:** Assign conflicts that are related to the specialist.
        **Output Format:** 
        1. **Specialists to Consult:** 
        - Specialist 1: [Reason for consultation] 
        - Specialist 2: [Reason for consultation] 
        2. **Assign Conflicts:**
        - Specialists 1: [Conflict numbers related to Specialist 1]
        - Specialists 2: [Conflict numbers related to Specialist 2]
        '''
        conflicts_assigned = agent_GP.chat(recuit_MDT_prompt)
        print(conflicts_assigned)
        
        DATA_COMPILER_INSTRUCT = "You are a helpful data organizer, your job is to convert the doctor's spoken content into structured data."
        agent_data_compiler = Agent(DATA_COMPILER_INSTRUCT, 'compiler')
        assign_conflicts_match = re.search(r"Assign Conflicts.*", conflicts_assigned, re.DOTALL)
        assign_conflicts = assign_conflicts_match.group(0).strip() if assign_conflicts_match else ""
        # print(assign_conflicts)
        assign_conflict_prompt = f'''Now that several doctors are being assigned conflict-specific discussion tasks, here are the assignment results <{assign_conflicts}>. 
        Please organize the distribution results into list format like this ["Specialist: conflict numbers", "Specialist: conflict numbers"].
        Only output the list.
        '''
        result = agent_data_compiler.chat(assign_conflict_prompt)
        assign_list = getlist(result)
        conflicts_dict = get_conflict_dict(assign_list)
        # print(conflicts_dict)
        specialists_list = list(conflicts_dict.keys())
        #Recruitment
        RESTRICT_INSTRUCT = ''
        agent_specialists = create_agent_specialists(specialists_list, RESTRICT_INSTRUCT)
        # print(specialists_list)
        conflict_to_specialists = {}
        for specialist, conflict_item in conflicts_dict.items():
            for conflict in conflict_item:
                if conflict in conflict_to_specialists:
                    conflict_to_specialists[conflict].append(specialist)
                else:
                    conflict_to_specialists[conflict] = [specialist]
        single_chat_list = {}
        multi_chat_list = {}
        for conflict_idx, specialist_item in conflict_to_specialists.items():
            if len(specialist_item) == 1:
                single_chat_list[conflict_idx] = specialist_item
            else:
                multi_chat_list[conflict_idx] = specialist_item

        print("0215----1:", multi_chat_list, single_chat_list)
        advices_from_single = generate_advices_from_single(single_chat_list, agent_specialists, conflicts_splited, goal_rel_med, goals, patient_condition)
        advices_from_multi, advices_from_multi_rounds, advices_from_multi_text = generate_advices_from_multi(multi_chat_list, agent_specialists, conflicts_splited, goal_rel_med, goals, patient_condition)
        specialist_advices = advices_from_single + advices_from_multi
        conflict_revise_res = conflict_revise(specialist_advices, agent_GP, "analyze")
        
        final_output = {
            "case_num": index+1,
            "patient_condition": patient_condition,
            "initial_conflicts": conflicts,
            "clinical_goals": goals,
            "initial_med_list": goal_rel_med,
            "New_plan_by_GP": new_conf,
            "MDT_disc_plan": conflicts_assigned,
            "second_medicine": conflict_revise_res,
            "first_discussion_group_medicine": { 
                "advices_from_single": advices_from_single, 
                "advices_from_multi": advices_from_multi,
                "single_chat_list": single_chat_list, 
                "multi_chat_list": multi_chat_list,
                "advices_from_multi_rounds": advices_from_multi_rounds, 
                "advices_from_multi_text": advices_from_multi_text, 
                }
        }

        output_path = f"/cases_results/qwen.json"
        new_data = final_output  
        try:
            with open(output_path, "r", encoding="utf-8") as file:
                data = json.load(file)  
        except FileNotFoundError:
            data = []
        if isinstance(data, list):
            data.append(new_data)  
        else:
            raise ValueError("JSON error")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print("done.")


