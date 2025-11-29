class EmergencyLevel:
    """
    Sistema de classificação de nível de emergência baseado no diagnóstico.
    Define níveis: VERDE, AMARELO, LARANJA, VERMELHO
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
        Retorna o nível de emergência para um diagnóstico.
        
        Args:
            diagnosis: diagnóstico predito
            confidence: confiança da predição (0-1)
        
        Returns:
            dict com informações do nível de emergência
        """
        level_key = cls.DIAGNOSIS_MAPPING.get(diagnosis, 'AMARELO')
        level_info = cls.LEVELS[level_key].copy()
        level_info['level'] = level_key
        
        # Ajustar recomendação baseado na confiança
        if confidence and confidence < 0.6:
            level_info['aviso'] = f'⚠️ Baixa confiança ({confidence:.0%}) - CONSULTE UM MÉDICO PARA CONFIRMAÇÃO'
        
        return level_info
    
    @classmethod
    def get_all_levels(cls):
        """Retorna todos os níveis disponíveis."""
        return cls.LEVELS
    
    @classmethod
    def add_diagnosis_mapping(cls, diagnosis, level):
        """Adiciona novo mapeamento diagnóstico -> nível."""
        if level not in cls.LEVELS:
            raise ValueError(f'Nível inválido: {level}')
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