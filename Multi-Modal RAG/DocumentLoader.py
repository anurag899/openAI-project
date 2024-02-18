from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import pandas as pd
from os import path 
import tiktoken
import os
import logging

class DocumentExtract():
    def __init__(self,endpoint,keys,model_id):
        self.endpoint = endpoint
        self.keys = keys
        self.model_id = model_id
    
    def load(self,document):
        f = open(document,'rb').read()
        return f
    
    def extract_text(self,document):
        form_recognizer_client = DocumentAnalysisClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.keys))
        f = self.load(document)
        result = form_recognizer_client.begin_analyze_document(self.model_id,f).result()
        return result
    
    def _get_cordinate(self,br):
        pageNumber = br.page_number
        x_min,y_min = list(br.polygon[0])
        x_max,y_max = list(br.polygon[2])
        return pageNumber,[x_min,x_max,y_min,y_max]

    def to_dataframe(self,result):
        data = pd.DataFrame(columns=['content','page_number','xmin','ymin','xmax','ymax'])
        for para in result.paragraphs:
            content = para.content
            pageNumber,coordinate = self._get_cordinate(para.bounding_regions[0])
            x_min,x_max,y_min,y_max = coordinate
            data.loc[len(data)] = [content,pageNumber,x_min,y_min,x_max,y_max]
        return data
    
    def extract_table(self,table):
        cells_list = []
        page_set = set()
        for cell in table.cells:
            row, col, content = cell.row_index, cell.column_index,cell.content
            boundingBox = cell.bounding_regions[0].polygon
            page_number = cell.bounding_regions[0].page_number
            page_set.add(page_number)
            cells_list.append((row,col,content,boundingBox))
        num_rows = max(cell[0] for cell in cells_list) + 1  
        num_cols = max(cell[1] for cell in cells_list) + 1  
        matrix = [[''] * num_cols for _ in range(num_rows)] 
        coordinates = []
        for cell in cells_list:
            row, col, content,boundingBox = cell    
            for idx in range(len(boundingBox)):
                coordinates.append((boundingBox[idx].x,boundingBox[idx].y))
            matrix[row][col] = content     
        x_coords, y_coords = zip(*coordinates)  # Unzip the coordinates into separate lists
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        table_dict = {}
        table_dict['table'] = matrix
        table_dict['coordinate'] = {'x-min':x_min,'y-min':y_min,'x-max':x_max,'y-max':y_max}
        table_dict['page_number'] = list(page_set)
        return table_dict
        

class DocumentChunk(DocumentExtract):
    def __init__(self,endpoint,keys,model_id):
        super().__init__(endpoint,keys,model_id)
        
    @classmethod
    def load_keys(cls,endpoint,keys,model_id):
        return cls(endpoint,keys,model_id)

    def get_token_length(self,content):
        content_encode = self.enc.encode(content)
        return len(content_encode)
    
    def filter_rows(self,dataframe,no_rows=5):
        data = dataframe.copy()
        data['token_len'] = data['content'].apply(self.get_token_length)
        return data[data['token_len'] > no_rows].reset_index(drop=True)
    
    def update_table_info(self,dataframe,table_dict):
        data = dataframe.copy()
        if len(table_dict['page_number']) > 1:
            raise ValueError('Page Number has 2 values')
        page_number = table_dict['page_number'][0]
        page_df = data[data.page_number==page_number]
        page_df = self._get_table_idx(page_df,table_dict)
        idx = list(page_df.index)
        idx_s,idx_e = idx[0],idx[-1]
        table_content = '\n'.join([';'.join(rows) for rows in table_dict['table']])
        table_content = f"\n<table-start> \n\n {table_content} \n\n <table-end>\n"
        row_dict = {'content':table_content,'page_number':page_number,
                   'xmin':table_dict['coordinate']['x-min'],
                   'ymin':table_dict['coordinate']['y-min'],
                   'xmax':table_dict['coordinate']['x-max'],
                   'ymax':table_dict['coordinate']['y-max']}
        df_r = pd.concat([data.iloc[:idx_s], pd.DataFrame(row_dict, index=[idx_s]), data.iloc[idx_e+1:]], ignore_index=True)
        return df_r

    def get_dataframe(self,filename,token_threshold,len_threshold=5):
        result = self.extract_text(filename)
        dataframe = self.to_dataframe(result)
        tables = result.tables
        data_update = dataframe.copy()
        logging.info(f'Total Identify Tables - {len(tables)}')
        for idx,table in enumerate(tables):
            table_dict = self.extract_table(table)
            data_update = self.update_table_info(data_update,table_dict)
        data_update['len_content'] = data_update['content'].apply(lambda x:len(x.split(' ')))
        data_update = data_update[data_update['len_content']>=len_threshold]
        self.enc = tiktoken.get_encoding("r50k_base")   
        data_update['token_len'] = data_update['content'].apply(self.get_token_length)
        doc_df = pd.DataFrame(columns=['content','page_number'])
        for page_num in data_update['page_number'].unique():
            appended_text = []
            page_df = data_update[data_update['page_number'] == page_num]
            current_len,current_text = 0,''
            for index, row in page_df.iterrows(): #TO-DO - Append bounding box while merging into paragphas
                if current_len + row['token_len'] <= token_threshold:
                    current_text += row['content'] + '\n'
                    current_len += row['token_len']
                else:
                    appended_text.append(current_text.strip())
                    current_text = row['content'] + '\n'
                    current_len = row['token_len']
            appended_text.append(current_text.strip())    
            if len(appended_text)>1:
                for text in appended_text:
                    doc_df.loc[len(doc_df)] = [text,page_num]
            else:
                doc_df.loc[len(doc_df)] = [appended_text[0],page_num]
        return doc_df  
    
    def isOverlap(self,content,big_rect):
        if (content["xmin"] <= big_rect['coordinate']["x-max"] and
            content["xmax"] >= big_rect['coordinate']["x-min"] and
            content["ymin"] <= big_rect['coordinate']["y-max"] and
            content["ymax"] >= big_rect['coordinate']["y-min"]):
                return True
        return False
    
    def _get_table_idx(self,page_df,table_dict):
        return page_df[page_df.apply(self.isOverlap,big_rect=table_dict,axis=1)]