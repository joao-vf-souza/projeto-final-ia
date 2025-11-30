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
    DIAGNOSIS_MAPPING = {
        'Gripe': 'AMARELO',
        'COVID-19': 'AMARELO',
        'Pneumonia': 'LARANJA',
        'Bronquite': 'AMARELO',
        'Apendicite': 'VERMELHO',
        'Gastroenterite': 'AMARELO',
        'Resfriado': 'VERDE',
        'Enxaqueca': 'VERDE',
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