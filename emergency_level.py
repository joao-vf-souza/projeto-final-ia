class EmergencyLevel:
    """
    Sistema de classificação de nível de emergência médica.
    
    Classifica diagnósticos em 4 níveis de urgência:
    - VERDE: Baixa urgência (consulta em dias)
    - AMARELO: Urgência moderada (consulta em horas)
    - LARANJA: Emergência (atendimento imediato)
    - VERMELHO: Emergência crítica (ambulância/192)
    """
    
    LEVELS = {
        'VERDE': {
            'color': '🟢',
            'descricao': 'Emergência Baixa',
            'acao': 'Consultar em dias',
            'recomendacao': 'Procure um posto de saúde durante horário comercial',
            'urgencia': 1
        },
        'AMARELO': {
            'color': '🟡',
            'descricao': 'Urgência',
            'acao': 'Consultar em poucas horas',
            'recomendacao': 'Procure o pronto-socorro ou UPA em poucas horas',
            'urgencia': 2
        },
        'LARANJA': {
            'color': '🟠',
            'descricao': 'Emergência',
            'acao': 'Procurar pronto-socorro hoje',
            'recomendacao': 'Vá ao pronto-socorro/ER o mais rápido possível',
            'urgencia': 3
        },
        'VERMELHO': {
            'color': '🔴',
            'descricao': 'Emergência Crítica',
            'acao': 'Ambulância/ER imediato',
            'recomendacao': 'LIGUE 192 (ambulância) IMEDIATAMENTE',
            'urgencia': 4
        }
    }
    
    # Mapeamento diagnóstico -> nível de emergência
    # VERDE: Baixa urgência | AMARELO: Urgência moderada | LARANJA: Emergência | VERMELHO: Crítica
    DIAGNOSIS_MAPPING = {
        # VERMELHO - Emergências Críticas (Risco de vida imediato)
        'heart attack': 'VERMELHO',
        'heart failure': 'VERMELHO',
        'sepsis': 'VERMELHO',
        'acute pancreatitis': 'VERMELHO',
        'acute kidney injury': 'VERMELHO',
        'gastrointestinal hemorrhage': 'VERMELHO',
        'sickle cell crisis': 'VERMELHO',
        'anaphylaxis': 'VERMELHO',
        'stroke': 'VERMELHO',
        
        # LARANJA - Emergências (Atendimento urgente necessário)
        'appendicitis': 'LARANJA',
        'pneumonia': 'LARANJA',
        'acute bronchiolitis': 'LARANJA',
        'acute bronchospasm': 'LARANJA',
        'cholecystitis': 'LARANJA',
        'diverticulitis': 'LARANJA',
        'pelvic inflammatory disease': 'LARANJA',
        'angina': 'LARANJA',
        'gallstone': 'LARANJA',
        'acute sinusitis': 'LARANJA',
        'cystitis': 'LARANJA',
        'urinary tract infection': 'LARANJA',
        'cornea infection': 'LARANJA',
        'otitis media': 'LARANJA',
        'threatened pregnancy': 'LARANJA',
        'problem during pregnancy': 'LARANJA',
        'hyperemesis gravidarum': 'LARANJA',
        'drug reaction': 'LARANJA',
        'concussion': 'LARANJA',
        'injury to the trunk': 'LARANJA',
        
        # AMARELO - Urgência Moderada (Consulta em horas)
        'asthma': 'AMARELO',
        'chronic obstructive pulmonary disease (copd)': 'AMARELO',
        'acute bronchitis': 'AMARELO',
        'infectious gastroenteritis': 'AMARELO',
        'noninfectious gastroenteritis': 'AMARELO',
        'esophagitis': 'AMARELO',
        'anxiety': 'AMARELO',
        'panic disorder': 'AMARELO',
        'depression': 'AMARELO',
        'gout': 'AMARELO',
        'hypertensive heart disease': 'AMARELO',
        'hypoglycemia': 'AMARELO',
        'strep throat': 'AMARELO',
        'conjunctivitis': 'AMARELO',
        'otitis externa (swimmer\'s ear)': 'AMARELO',
        'croup': 'AMARELO',
        'vaginitis': 'AMARELO',
        'pain after an operation': 'AMARELO',
        'injury to the arm': 'AMARELO',
        'injury to the leg': 'AMARELO',
        'sprain or strain': 'AMARELO',
        'herniated disk': 'AMARELO',
        'spontaneous abortion': 'AMARELO',
        'pyogenic skin infection': 'AMARELO',
        'ear drum damage': 'AMARELO',
        'obstructive sleep apnea (osa)': 'AMARELO',
        'sinus bradycardia': 'AMARELO',
        
        # VERDE - Baixa Urgência (Consulta em dias)
        'common cold': 'VERDE',
        'allergy': 'VERDE',
        'seasonal allergies (hay fever)': 'VERDE',
        'conjunctivitis due to allergy': 'VERDE',
        'eczema': 'VERDE',
        'psoriasis': 'VERDE',
        'contact dermatitis': 'VERDE',
        'actinic keratosis': 'VERDE',
        'skin pigmentation disorder': 'VERDE',
        'skin polyp': 'VERDE',
        'sebaceous cyst': 'VERDE',
        'vaginal cyst': 'VERDE',
        'diaper rash': 'VERDE',
        'stye': 'VERDE',
        'dental caries': 'VERDE',
        'gum disease': 'VERDE',
        'chronic back pain': 'VERDE',
        'chronic constipation': 'VERDE',
        'hemorrhoids': 'VERDE',
        'rectal disorder': 'VERDE',
        'benign prostatic hyperplasia (bph)': 'VERDE',
        'idiopathic excessive menstruation': 'VERDE',
        'idiopathic irregular menstrual cycle': 'VERDE',
        'idiopathic painful menstruation': 'VERDE',
        'eustachian tube dysfunction (ear disorder)': 'VERDE',
        'nose disorder': 'VERDE',
        'arthritis of the hip': 'VERDE',
        'bursitis': 'VERDE',
        'carpal tunnel syndrome': 'VERDE',
        'degenerative disc disease': 'VERDE',
        'spinal stenosis': 'VERDE',
        'spondylosis': 'VERDE',
        'peripheral nerve disorder': 'VERDE',
        'brachial neuritis': 'VERDE',
        'complex regional pain syndrome': 'VERDE',
        'macular degeneration': 'VERDE',
        'hiatal hernia': 'VERDE',
        'liver disease': 'VERDE',
        'fungal infection of the hair': 'VERDE',
        'multiple sclerosis': 'VERDE',
        'schizophrenia': 'VERDE',
        'personality disorder': 'VERDE',
        'developmental disability': 'VERDE',
        'marijuana abuse': 'VERDE',
        'vulvodynia': 'VERDE',
        'temporary or benign blood in urine': 'VERDE',
    }
    
    @classmethod
    def get_level(cls, diagnosis, confidence=None):
        """
        Retorna o nível de emergência para um diagnóstico específico.
        
        Parâmetros:
            diagnosis (str): Nome do diagnóstico/doença
            confidence (float, optional): Confiança da predição entre 0 e 1
        
        Retorna:
            dict: Dicionário contendo informações do nível de emergência:
                - level: código do nível (VERDE, AMARELO, LARANJA, VERMELHO)
                - color: emoji do nível
                - descricao: descrição do nível
                - acao: ação recomendada
                - recomendacao: orientação detalhada
                - urgencia: valor numérico de urgência (1-4)
                - aviso: mensagem de alerta se confiança baixa
        """
        # Obtém o nível de emergência do diagnóstico (padrão: AMARELO se não encontrado)
        level_key = cls.DIAGNOSIS_MAPPING.get(diagnosis, 'AMARELO')
        level_info = cls.LEVELS[level_key].copy()
        level_info['level'] = level_key
        
        # Adiciona aviso se a confiança da predição for baixa
        if confidence and confidence < 0.6:
            level_info['aviso'] = f'Baixa confiança ({confidence:.0%}) - CONSULTE UM MÉDICO PARA CONFIRMAÇÃO'
        
        return level_info
    
    @classmethod
    def get_all_levels(cls):
        """
        Retorna todos os níveis de emergência disponíveis no sistema.
        
        Retorna:
            dict: Dicionário com todos os níveis e suas informações
        """
        return cls.LEVELS
    
    @classmethod
    def add_diagnosis_mapping(cls, diagnosis, level):
        """
        Adiciona ou atualiza o mapeamento de um diagnóstico para um nível de emergência.
        
        Parâmetros:
            diagnosis (str): Nome do diagnóstico/doença
            level (str): Nível de emergência (VERDE, AMARELO, LARANJA ou VERMELHO)
        
        Raises:
            ValueError: Se o nível especificado não existir
        """
        if level not in cls.LEVELS:
            raise ValueError(f'Nível inválido: {level}. Use: VERDE, AMARELO, LARANJA ou VERMELHO')
        cls.DIAGNOSIS_MAPPING[diagnosis] = level


# Exemplo de uso
if __name__ == '__main__':
    # Teste
    print("Níveis de Emergência Disponíveis:")
    print("=" * 50)
    for level_key, info in EmergencyLevel.get_all_levels().items():
        print(f"{info['color']} {level_key}: {info['descricao']}")
        print(f"   Ação: {info['acao']}")
        print(f"   Recomendação: {info['recomendacao']}")
        print()
    
    print("\nExemplos de Diagnósticos:")
    print("=" * 50)
    for diagnosis, level in EmergencyLevel.DIAGNOSIS_MAPPING.items():
        level_info = EmergencyLevel.get_level(diagnosis, 0.85)
        print(f"{level_info['color']} {diagnosis} -> {level}")